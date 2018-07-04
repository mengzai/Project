/**
  * Created by wenqiang on 2017/7/21.
  */
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import collection._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{ChiSqSelector, Normalizer, StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.json4s.jackson.Json
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object AladingModel1 {
  def numericalProcessing(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
    } catch {
      case ex: NumberFormatException => -999.0
    }
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def numericalProcessing(df: DataFrame, hiveCtx:HiveContext): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>numericalProcessing(r.getAs(x))).toSeq:_*))
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = hiveCtx.createDataFrame(data, schema)
    dfFigure
  }

  def preProcess(hiveCtx: HiveContext,df: DataFrame,ignoreFields: Seq[String],labelField: String) = {
    // 1.Field selection
    val featuresFields: Array[String] = df.columns.diff(labelField).diff(ignoreFields)
    val selectData: SchemaRDD = numericalProcessing(df.select(labelField,featuresFields.toSeq:_*),hiveCtx)
    val assembler = new VectorAssembler().setInputCols(featuresFields).setOutputCol("features")
    val mlData: SchemaRDD = assembler.transform(selectData).select(labelField,"features")

    mlData
  }

  def Sampling(hiveCtx: HiveContext,data: DataFrame,positiveProbability: Double,negativeProbability: Double,labelName: String = "label"): SchemaRDD = {
    val classData: RDD[(Int, Row)] = data.map(row => {
      if (row.getAs[Int](labelName) == 1)
        (row, 1)
      else (row, 2)
    }).map(x => (x._2, x._1))
    val fractions: Map[Int, Double] = (List((1, positiveProbability), (2, negativeProbability))).toMap
    val sampleData: RDD[(Int, Row)] = classData.sampleByKey(withReplacement = false, fractions, 0)
    val generatedDate: RDD[Row] = sampleData.map(x => x._2)
    println("************* Positive and negative sample balance *************")
    val numberClass: RDD[Int] = sampleData.map(x =>x._1)
    println(numberClass.filter(x =>x==1).count())
    println (numberClass.filter(x =>x==2).count())
    println("****************************************************************")

    val dfName = data.columns
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, IntegerType, true)))
    val dfSample = hiveCtx.createDataFrame(generatedDate, schema)
    dfSample

  }

  def RFModel(data: List[DataFrame], label: String) ={
    val labelField = label
    val dataTrainData = data(0)
    val dataTestData = data(1)

    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    rf.setMaxBins(32).setMaxDepth(6).setNumTrees(90).setMinInstancesPerNode(4).setImpurity("gini")
    //val gbt = new GBTClassifier().

    // feature
    val labelIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol(labelField)
      .setOutputCol("indexedLabel")
      .fit(dataTrainData)
    val featureIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(dataTrainData)
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    // Train model. This also runs the indexers.
    val model: PipelineModel = pipeline.fit(dataTrainData)
    // Make predictions.
    val predictions: SchemaRDD = model.transform(dataTestData)
    predictions.select("predictedLabel", "indexedLabel", "indexedFeatures").show(50)

    // ROC metrics
    val predictScores = predictions.select("predictedLabel", "indexedLabel").rdd
      .map(r => (r(0).toString.toDouble, r(1).toString.toDouble))
    val metrics = new BinaryClassificationMetrics(predictScores)
    val auROC = metrics.areaUnderROC
    println ("***** AUC metrics *****")
    println("Area under ROC = " + auROC)
  }

  def fieldDistinctCount(data: DataFrame): Array[(String, Long)] = {
    val fieldName: Array[String] = data.columns
    val distinctCount =data.agg(countDistinct(fieldName(0)),fieldName.drop(1).map(x=>countDistinct(x)).toSeq:_*)
    distinctCount.printSchema()
    println("****************** countDistinct running *****************")

    val distinctCountValue: Seq[Long] = distinctCount.first().toSeq.asInstanceOf[Seq[Long]]
    fieldName.zip(distinctCountValue)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("aladingModel")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("WARN")

    // 1.Load Data
    val rawTrainData = hiveCtx.sql(s"select * from test.alading_train_4_10_good_new where datadate<='2016-10-01'")
    val rawTestData = hiveCtx.sql(s"select * from test.alading_train_4_10_good_new where datadate>='2017-01-01'")
    println(s"***** Raw data structure  *****")
    rawTrainData.printSchema()

    // 1.2 Sampling
    //val rawTrainData: SchemaRDD = Sampling(hiveCtx,rawTrainData1,positiveProbability = 1.0,negativeProbability = 0.05)


    // 2.Preprocessor
    val labelField: String = s"label"
    val invalidField: List[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app","Recent_order_product","Recent_order_days","Recent_investment_days","Recent_investment_due_days")
    val singleValueField = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num <=1")
      .map(r => r.getAs[String](0)).collect().toSeq
    //val singleValueField: Array[String] = fieldDistinctCount(rawTestData).filter(x =>x._2>1).map(x=>x._1)
    val ignoreField: Seq[String] = invalidField ++ singleValueField
    val dataTrainData: SchemaRDD = preProcess(hiveCtx,rawTrainData,ignoreField,labelField)
    val dataTestData = preProcess(hiveCtx,rawTestData,ignoreField,labelField)

    // 3.Sampling
    //val sampldTrainData: SchemaRDD = Sampling(hiveCtx,dataTrainData,positiveProbability = 1.0,negativeProbability = 0.05)
    val sampldTrainData = dataTrainData
    val sampldTestData = dataTestData

    // 4.ModelBuilder
    RFModel(List(sampldTrainData,sampldTestData), labelField)

    // 5.The result of running by day
       /*
    val ecifTime: RDD[(String, String)] =
      hiveCtx.sql(s"select ecif_id,datadate from test.alading_train_4_10_new where datadate>='2017-01-01'")
        .map(r=>(r.getAs(0).toString,r.getAs(1).toString))
    val preData: RDD[Row] = ecifTime.zip(preLabel).map(r=>Row(r._1._1,r._1._2,r._2._1.toString,r._2._2.toString))

    val preName: Array[String] = Array("ecif_id","datadate","predict","realLabel")
    val schema = StructType(preName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfPreLabel = hiveCtx.createDataFrame(preData, schema)
    dfPreLabel.show()
    // insert into hive
    dfPreLabel.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_pre_out as select * from tmp_table")
    */
  }
}

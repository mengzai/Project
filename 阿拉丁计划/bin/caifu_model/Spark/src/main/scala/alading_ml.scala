package cn.creditease.invest.test
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
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.linalg
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

object alading_ml {
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

  def preProcess(hiveCtx: HiveContext,df: DataFrame,featuresFields: Array[String],labelField: String,indexSample: Int = 0): SchemaRDD = {
    // 1.Field selection
    var selectData: SchemaRDD = numericalProcessing(df.select(labelField,featuresFields.toVector.toSeq:_*),hiveCtx)
    print("****************pre***************")

    // 2.sample
    if (indexSample == 1){
      selectData = Sampling(hiveCtx,selectData,positiveProbability = 1.0 ,negativeProbability=0.05)
    }
    // 3. ml dataType
    val assembler = new VectorAssembler().setInputCols(featuresFields).setOutputCol("features")
    val mlData: SchemaRDD = assembler.transform(selectData).select(labelField,"features")

    mlData
  }

  def Sampling(hiveCtx: HiveContext,data: DataFrame,positiveProbability: Double,negativeProbability: Double,labelName: String = "label"): SchemaRDD = {
    val classData: RDD[(Int, Row)] = data.map(row => {
      if (row.getAs[Double](labelName) == 1.0)
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
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfSample = hiveCtx.createDataFrame(generatedDate, schema)
    dfSample
  }

  def Feature_important(rfModel:RandomForestClassificationModel,featuresFields:Array[String],sc:SparkContext,hiveCtx: HiveContext)={

    println("#####Feature_important    start#######")
    val Important: linalg.Vector =rfModel.featureImportances
    println(Important)
    val featureIM: Array[String] =Important.toArray.map(x=>x.toString)

    val dataTranspose: Array[(String, String)] =(featuresFields  zip featureIM ).toSeq.toArray

    val dfName: Array[String] = Array("name","important")
    val data: RDD[Row] = sc.parallelize(dataTranspose.map(r => Row(r._1,r._2)).toVector)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))

    val dfDescrOut: SchemaRDD = hiveCtx.createDataFrame(data, schema)
    dfDescrOut.show()
//
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_Feature_important_new as select * from tmp_table")
  }

  def GBDT_ml(data: List[DataFrame], label: String, featuresFields:Array[String],sc:SparkContext,hiveCtx:HiveContext,features: String = "features")={

    val dataTrainData = data(0)
    val dataTestData = data(1)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(dataTrainData)


    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model.  This also runs the indexer.
    val model = pipeline.fit(dataTrainData)

    // ROC metrics
    val predictions_train = model.transform(dataTrainData)
    // ROC metrics
    val predictScores_train = predictions_train.select("probability", "label").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics_train = new BinaryClassificationMetrics(predictScores_train)
    val auROC_train = metrics_train.areaUnderROC
    println("GBDT 训练集  AUC" + auROC_train)

    //测试集
    val predictions = model.transform(dataTestData)
    val predictScores = predictions.select("probability", "label").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics = new BinaryClassificationMetrics(predictScores)
    val auROC = metrics.areaUnderROC
    println("GBDT 测试集  AUC" + auROC)
    predictions
  }

  def RFModel(data: List[DataFrame], label: String, featuresFields:Array[String],sc:SparkContext,hiveCtx:HiveContext,features: String = "features"): SchemaRDD ={
    val dataTrainData: SchemaRDD = data(0)
    val dataTestData = data(1)

    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("label_idx")
      .fit(dataTrainData)

    val featureIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol(features)
      .setOutputCol("features_idx")
      .fit(dataTrainData)

    val RF = new RandomForestClassifier().setLabelCol("label_idx").setMaxBins(32).
      setMaxDepth(12).setNumTrees(100).setMinInstancesPerNode(8).setImpurity("gini")
       //val GBT = new GBTClassifier().setLabelCol("label_idx").setMaxBins(32)
//    .setMaxDepth(6).setMinInstancesPerNode(4).setImpurity("gini").setMaxIter(10)

    // Train model. This al so runs the indexers.

    val model: RandomForestClassificationModel = RF.fit(featureIndexer.transform(indexer.transform(dataTrainData)))

    Feature_important(model,featuresFields,sc,hiveCtx)

    //val model: GBTClassificationModel = GBT.fit(featureIndexer.transform(indexer.transform(dataTrainData)))
    // Make predictions.
    val predictions_train = model.transform(dataTrainData)

    val converter = new IndexToString()
      .setInputCol("probabilication")
      .setOutputCol("label_new")
      .setLabels(indexer.labels)

    // ROC metrics
    val predictScores_train = predictions_train.select("probability", "label_new").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics_train = new BinaryClassificationMetrics(predictScores_train)
    val auROC_train = metrics_train.areaUnderROC
    println("RF 训练集  AUC" + auROC_train)

    //测试集
    val predictions = model.transform(dataTestData)
    val predictScores = predictions.select("probability", "label").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics = new BinaryClassificationMetrics(predictScores)
    val auROC = metrics.areaUnderROC
    println("RF 测试集  AUC" + auROC)
    predictions
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
    sc.setLogLevel("ERROR")

    // 1.Load Data
    val rawTrainData = hiveCtx.sql(s"select * from test.alading_alldata_end_5 where datadate<='2016-10-01'")
    val rawTestData = hiveCtx.sql(s"select * from test.alading_alldata_end_5 where datadate>='2017-04-01'")
    println(s"***** Raw data structure  *****")
//    rawTrainData.printSchema()

    // 2.Preprocessor
    val labelField: String = "label"
    val invalidField: List[String] = List("label","phone","id_number","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app","recent_order_product","recent_order_days","recent_investment_days","recent_investment_due_days")
    val singleValueField = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num <=1")
      .map(r => r.getAs[String](0)).collect().toSeq
    //val singleValueField: Array[String] = fieldDistinctCount(rawTestData).filter(x =>x._2>1).map(x=>x._1)
    val ignoreField: Seq[String] = invalidField ++ singleValueField

    val featuresFields: Array[String] = rawTrainData.columns.diff(labelField).diff(ignoreField)

    val dataTrainData: SchemaRDD = preProcess(hiveCtx,rawTrainData,featuresFields,labelField,indexSample = 1)
    val dataTestData = preProcess(hiveCtx,rawTestData,featuresFields,labelField)

    // 3.ModelBuilder
    val metricsResult_RF: SchemaRDD = RFModel(List(dataTrainData,dataTestData), labelField,featuresFields,sc,hiveCtx)

    val metricsResult_GBDT: SchemaRDD = GBDT_ml(List(dataTrainData,dataTestData), labelField,featuresFields,sc,hiveCtx)

//    // 5.The result of running by day
//    val metricsResult1: SchemaRDD = metricsResult_RF.select("label","probability","prediction")

    println("######## 保存结果")
    // 5.The result of running by day
    val metrics: SchemaRDD = metricsResult_RF.select("label","prediction","probability")
    val metricsData: RDD[(String, String, String)] = metrics.map(x=>(x(0).toString,x(1).toString,x(2).toString.split(",")(0)))
    val ecifidData: RDD[(String, String)] = rawTestData.select("ecif_id","datadate").map(x=>(x(0).toString,x(1).toString))
    val outData: RDD[Row] = ecifidData.zip(metricsData).map(x=>Row(x._1._1,x._1._2,x._2._1,x._2.x._2,x._2._3))

    val dfName = Array("ecif_id","datadate","label","prediction","probability")
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val metricsDf = hiveCtx.createDataFrame(outData, schema)
    metricsDf.registerTempTable("metricsResult")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_day_result_ori as select * from metricsResult")

    println("######## 保存结果 end")
    hiveCtx.sql("create table alading_ecifidData as select * from ecifidData")
  }
}



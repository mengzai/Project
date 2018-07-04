package cn.creditease.invest.test

import cn.creditease.invest.test.SparkTest.GBDT_model
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.StandardScalerModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.tree.{GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.util.random.RandomSampler

import collection._
import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.ml.feature._
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}

import scala.collection.JavaConverters._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg.Matrix
import org.apache.hadoop.io.{NullWritable, Text}
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg

import scala.collection.mutable.Map
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{ChiSqSelector, Normalizer, StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg
//import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
//import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
//import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.json4s.jackson.Json
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier

/**
  * Created by gangfang6 on 2017/7/14.
  */
object line_down_spark_test {

  def myToDouble(x: Any) = x match{
    case s: String => try{s.toDouble}catch{case ex: NumberFormatException => -999.0}
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }


  def Feature_coverage(df:DataFrame,sc:SparkContext)= {
    val all_length=df.count()
    val buf = new ListBuffer[Any]
    for (x: String <- df.columns) {

      val ds =df.describe(x)
      ds.show()

      val result: RDD[Any] = ds.toDF().rdd.map(it => {
        it.get(it.fieldIndex(x))
      })

      val t: Array[Any] = result.collect()
      val Cnt: Int =df.describe(x).first().get(1).toString.toInt
      val cover=Cnt*1.0/all_length

      val result_vector=t.toVector
      println(x,Cnt,cover,t.toVector)
      val site =(x,Cnt,cover,t.toVector)
      buf.append(site)
    }

    //    存储文件
    sc.parallelize(buf.toVector)
      .repartition(1).saveAsTextFile("/user/hive/warehouse/db/test.db/alading_cover_test")
  }

  //应用信息增益选择特征
  def Use_inforGain_find_features(sc: SparkContext,df:DataFrame,topK:Int)={

    val myMap = new scala.collection.mutable.HashMap[String, Double]
    for (x: String <- df.columns) {
      println(x)
      val featureInd: Int = df.columns.indexOf(x)
      val targetInd: Int = df.columns.indexOf("label")
      val label_feature_RDD: RDD[(Double, Double)] =df.map(r => (myToDouble(r.getAs(targetInd)),myToDouble(r.getAs(featureInd))))
      val Inforgainrate =inforGainRate(sc,label_feature_RDD)
      print(x,Inforgainrate)
      myMap += (x -> Inforgainrate)
    }
//    val mapSortSmall = myMap.toList.sortBy(_._2)//从小到大
    val mapSortBig = myMap.toList.sortBy(-_._2)   //从大到小

    //返回前K个列名
    val Topk_vec: List[String] =mapSortBig.take(topK).map(x=>x._1)
    println("####################从大到小打印特征重要性   start######################")
    Topk_vec.foreach(println)
    println("####################从大到小打印特征重要性  end#########################")
    Topk_vec
  }



  def feature_extract(df: DataFrame,sc:SparkContext,feature_name_df:DataFrame) = {
    // index of target

    val targetInd: Int = df.columns.indexOf("label")

    // index of feat (exclude columns)
    val ignored: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app","Recent_investment_days","Recent_investment_due_days","Recent_order_days")

    val ignored_and_label: Seq[String] = List("ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app","Recent_investment_days","Recent_investment_due_days","Recent_order_days")

    val featInd: Array[Int] = df.columns.diff(ignored).map(df.columns.indexOf(_))

    val columnNames1: Array[String] =df.columns.diff(ignored_and_label)
    val taeget_data=df.select(columnNames1.map(c => col(c)): _*)
    //  labeledPoint
    val data_get=df.map(r => LabeledPoint(myToDouble(r.getAs(targetInd)), Vectors.dense(featInd.map(x=>myToDouble(r.getAs(x))))))
    (taeget_data,data_get)
  }

  //归一化
  def  Normalizer(df:RDD[LabeledPoint])={

    val l2 =  new Normalizer(2)
    l2.transform(df.map(x=>x.features))
    df.map(x =>LabeledPoint(x.label,x.features))
  }
  //
  //  def class_balance(dataset: DataFrame, label: String = "label"): DataFrame = {
  //    // Re-balancing (weighting) of records to be used in the logistic loss objective function
  //    val (datasetSize, positives) = dataset.select(count("*"), sum(dataset(label))).as[(Long, Double)].collect.head
  //    val balancingRatio = positives / datasetSize
  //
  //    val weightedDataset = {
  //      dataset.withColumn("classWeightCol", when(dataset(label) === 0.0, balancingRatio).otherwise(1.0 - balancingRatio))
  //    }
  //    weightedDataset
  //  }

  //StandardScaler
  def StandardScaler(df:RDD[LabeledPoint])= {
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(df.map(x =>x.features))
    val data_feature_Standard=df.map(r => (r.label, scaler2.transform(r.features)))
    data_feature_Standard.map(x =>LabeledPoint(x._1,x._2))
  }

  //get_train_test
  def train_model(training:RDD[LabeledPoint],test:RDD[LabeledPoint])={
    //Training
    //    LR
    //    val lr = new LinearRegressionWithSGD()
    //    lr.optimizer
    //      .setNumIterations(10)
    //      .setUpdater(new L1Updater)
    //      .setRegParam(0.01)
    //
    //    println("Start Training")
    //    val model = lr.run(training)
    val model=RandomForest.trainClassifier(training, 2, collection.immutable.Map[Int, Int](), 10, "auto", "gini", 6, 2, 123)
    val predictionAndLabels = test.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    // ROC Curve
    val roc = metrics.roc
    // AUROC
    val auROC = metrics.areaUnderROC

    println("3333333333333333##################################################################################")
    println("Area under ROC = " + auROC)
    println("3333333333333333###################################################################################")
  }

  def feature_select(data:RDD[LabeledPoint],TopK:Int)={
    val selector = new ChiSqSelector(TopK)
    //创建一个特征选择模型
    val transformer=selector.fit(data)
    //打印出被选特征的index，counting from 0
    print("打印出被选特征的index，counting from 0")
    print(transformer.selectedFeatures.mkString("~"))
    print("打印出被选特征的index，counting from 1")
    //选取top 50的特征
    data.map{lp =>
      LabeledPoint(lp.label,transformer.transform(lp.features))
    }
  }

  //  def evaluate_model(model: _root_.org.apache.spark.mllib.tree.model.RandomForestModel,test:RDD[LabeledPoint])={
  //    val predictionAndLabels = test.map{ case LabeledPoint(label, features) =>
  //      val prediction = model.predict(features)
  //      (prediction, label)
  //    }
  //    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
  //    // ROC Curve
  //    val roc = metrics.roc
  //    // AUROC
  //    val auROC = metrics.areaUnderROC
  //
  //    println("3333333333333333##################################################################################")
  //    println("Area under ROC = " + auROC)
  //    println("3333333333333333###################################################################################")
  //  }

  //
  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  //信息熵
  def inforEntropy(target_attribute: Array[Double]): Double = {
    val temp = new scala.collection.mutable.HashMap[Double, Int]
    for (item <- target_attribute) {
      if (temp.contains(item)) {
        temp(item) = temp(item) + 1
      } else {
        temp.put(item, 1)
      }
    }
    var Entropy: Double =0.0
    for (item <- temp) {
      Entropy += (-1) * (item._2.toDouble / target_attribute.length) * log2(item._2.toDouble / target_attribute.length)
    }
    Entropy
  }

  //计算特征与目标特征之间的信息增益
  def inforGain(sc: SparkContext, feature_attribute: RDD[(Double, Double)]): (Double, Double) = {
    val target = feature_attribute.map { x => x._2 }.toArray()
    val Entropy1 = inforEntropy(target)
    val all_Entropy = sc.accumulator(0.0)
//    feature_attribute.groupBy(x => x._1).foreach { x => all_Entropy += (x._2.size.toDouble / target.length) * inforEntropy(x._2.map(x => x._2).toArray)}
    val X = feature_attribute.map { x => x._1 }
    val Y = feature_attribute.map { x => x._2 }
    val correlation: Double = Statistics.corr(X, Y, "pearson")
    println(Entropy1)
    println(all_Entropy.value)
    ((Entropy1 - all_Entropy.value), correlation)
  }

  //计算特征与目标特征之间的信息增益率
  def inforGainRate(sc: SparkContext, feature_attribute: RDD[(Double, Double)]) = {
    val target = feature_attribute.map { x => x._2 }.toArray()
    val Entropy1 = inforEntropy(target)
    val all_Entropy = sc.accumulator(0.0)
    feature_attribute.groupBy(x => x._1).collect().foreach(x=>all_Entropy+=(x._2.size.toDouble*1.0/target.length)*inforEntropy(x._2.map(x=>x._2).toArray))

    val X = feature_attribute.map { x => x._1 }
    val Y = feature_attribute.map { x => x._2 }
    val correlation: Double = Statistics.corr(X, Y, "pearson")

    /*    // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
72     // If a method is not specified, Pearson's method will be used by default.
73     val correlMatrix: Matrix = Statistics.corr(data, "pearson")*/
    println(all_Entropy.value)
    (Entropy1 - all_Entropy.value).toDouble/inforEntropy(X.toArray())
  }

  case class Person(name:String,col1:Int,col2:String)
  case class Final_result(name: String, cnt: String,cover_rate: String,mean: String,min:String,max: String,std: String)
  def Parse(line: String) = {
    val pieces = line.stripPrefix("(").stripSuffix("))").split("Vector")
    val id1 = pieces(0).split(",")
    val id2 = pieces(1).split(",")
    Final_result(id1(0).toString,id1(1).toString,id1(2).toString,id2(1).toString,id2(3).toString,id2(4).toString,id2(2).toString)
  }

  def get_the_list_to_excle(sc :SparkContext)= {
    val path = "/user/hive/warehouse/db/test.db/alading_cover_train/part-00000"
    val rawblocks = sc.textFile(path)
    val parsed: RDD[Final_result] = rawblocks.map(x => Parse(x))
    parsed.foreach(println)
    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import hiveContext.implicits._
    hiveContext.sql("use test")
    val hive_df = parsed.toDF.registerTempTable("alading_cover_train")
    print("############end###################")
  }
  def GBDT_model(trainingData:RDD[LabeledPoint],testData:RDD[LabeledPoint])={

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")//default param
    boostingStrategy.numIterations ->100
    boostingStrategy.learningRate ->0.01
    boostingStrategy.validationTol->0.001
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo -> scala.collection.mutable.Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error

    val train_labelAndPreds = trainingData.map { point =>
      val prediction = model.predict(point.features)
      println(prediction)
      (point.label, prediction)
    }


    val metrics_train = new BinaryClassificationMetrics(train_labelAndPreds)
    // ROC Curve
    val roc_train = metrics_train.roc
    // AUROC
    val auROC_train = metrics_train.areaUnderROC

    println("###########训练集 ##################################################################################")
    println(" GBDT 训练集  Area under ROC = " + auROC_train)

    val test_predictionAndLabels = testData.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    // ROC Curve
    val roc = metrics.roc
    // AUROC
    val auROC = metrics.areaUnderROC
    println("############测试集###################################################################################")
    println(" GBDT 测试集   Area under ROC = " + auROC)


    //    val testErr = test_predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / testData.count()
    //    println("Test Error = " + testErr)
    //    println("Learned classification GBT model:\n" + model.toDebugString)

    // Save and load model
    //    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    //    val sameModel = GradientBoostedTreesModel.load(sc,
    //      "target/tmp/myGradientBoostingClassificationModel")
  }


  def Feature_important(model:PipelineModel,featuresFields:Array[String],sc:SparkContext,hiveCtx: HiveContext)={

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]

    println("start")
    val Important: linalg.Vector =rfModel.featureImportances
    val featureIM: Array[String] =Important.toArray.map(x=>x.toString)
    val dataTranspose: Array[(String, String)] =(featuresFields  zip featureIM ).toSeq.toArray

    val dfName: Array[String] = Array("name","important")
    val data: RDD[Row] = sc.parallelize(dataTranspose.map(r => Row(r._1,r._2)).toVector)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))

    val dfDescrOut: SchemaRDD = hiveCtx.createDataFrame(data, schema)
    dfDescrOut.show()

    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_Feature_important as select * from tmp_table")

  }

  def RF_model(sc:SparkContext,hiveCtx:HiveContext,labelField:String,dataTrainData:SchemaRDD,dataTestData:SchemaRDD,featuresFields_train:Array[String])={
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    rf.setMaxBins(32).setMaxDepth(6).setNumTrees(90).setMinInstancesPerNode(4).setImpurity("gini")

    // feature
    val labelIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol(labelField)
      .setOutputCol("indexedLabel")
      .fit(dataTrainData)

    val featureIndexer: VectorIndexerModel = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(dataTrainData)
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    // Train model. This also runs the indexers.
    val model: PipelineModel = pipeline.fit(dataTrainData)
    // Make predictions.

    Feature_important(model,featuresFields_train,sc,hiveCtx)
    val predictions = model.transform(dataTestData)
    predictions.select("predictedLabel", "indexedLabel", "indexedFeatures").show(5)

  }


  def numericalProcessing(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
    } catch {
      case ex: NumberFormatException => -999.0
    }
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def dfToDouble(df: DataFrame, hiveCtx:HiveContext): SchemaRDD = {
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
    val selectData: SchemaRDD = dfToDouble(df.select(labelField,featuresFields.toSeq:_*),hiveCtx)

    val assembler = new VectorAssembler().setInputCols(featuresFields).setOutputCol("features")
    val mlData: SchemaRDD = assembler.transform(selectData).select(labelField,"features")

    (mlData,featuresFields)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkTest").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // 得到数据集
//    val input_df=HiveCtx.sql(s"select * from test.alading_train_4_10_new where datadate>'2016-10-01'")

    //    input_df.printSchema()

    //特征提取+去空值+labeledPoint
    val sqlCtx = new SQLContext(sc)
    val hiveCtx = new HiveContext(sc)
    import sqlCtx.implicits._
    val path="./data/people.json"
    val input_df: DataFrame =sqlCtx.read.json(path)
    import sqlCtx.implicits._
    val targe_name =sqlCtx.read.json("./data/people.json")
    val (targer_df ,data_get)=feature_extract(input_df,sc,targe_name)


    //特征选择：覆盖率；信息增益
    //覆盖率选择特征
    //    Feature_coverage(targer_df,sc)//保存到/user/hive/warehouse/db/test.db/alading_cover_train/
    //对覆盖率进行提取特征
    // 信息增益选择特征
//    val Topk_vec: List[String] =Use_inforGain_find_features(sc,targer_df,100)

    //标准化
    //   val data_standard=StandardScaler(data_get)

    //卡方分布 选择topK
    //    val filteredData =feature_select(data_standard,100)

    //归一化
    //val train =Normalizer(data_standard)

    //模型randomSplit([0.7,0.3],123)
    val splits = targer_df.randomSplit(Array(0.7, 0.3), seed = 11L)

    val training: SchemaRDD = splits(0).cache()
    val test: SchemaRDD = splits(1)

    //训练模型
//    val model=train_model(training,test)
//    val model=GBDT_model(training,test)

    // 2.Preprocessor
    val labelField: String = s"label"

    //val singleValueField: Array[String] = fieldDistinctCount(rawTestData).filter(x =>x._2>1).map(x=>x._1)

    val ignoreFields: List[String] = List("label","ecif_id")
    val (dataTrainData,featuresFields_train) = preProcess(hiveCtx,training,ignoreFields,labelField)
    val (dataTestData,featuresFields_test) = preProcess(hiveCtx,test,ignoreFields,labelField)
    RF_model(sc,hiveCtx,labelField,dataTrainData,dataTestData,featuresFields_train)

  }
}

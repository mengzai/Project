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
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{ChiSqSelector, Normalizer, StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.json4s.jackson.Json

object AladingModela {
  def myToDouble(x: Any): Double = x match {
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
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>myToDouble(r.getAs(x))).toSeq:_*))
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = hiveCtx.createDataFrame(data, schema)
    dfFigure
  }

  def dfToDouble(df: DataFrame): RDD[LabeledPoint] = {
    df.map(r => LabeledPoint(myToDouble(r.getAs(0)),Vectors.dense(r.toSeq.drop(1).map(myToDouble).toArray)))
  }

  def myNormalizer(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val l2 = new Normalizer(2)
    l2.transform(data.map(x => x.features))
    data.map(x => LabeledPoint(x.label, x.features))
  }

  def myStandardScalerFit(data: RDD[LabeledPoint]): StandardScalerModel = {
    val scalar: StandardScalerModel = new StandardScaler(withMean = true, withStd = true).fit(data.map(r => r.features))
    scalar
  }

  def myStandardScalerTrans(data: RDD[LabeledPoint],scalar: StandardScalerModel) = {
    val dataOut: RDD[LabeledPoint] = data.map(lp => LabeledPoint(lp.label, scalar.transform(lp.features)))
    dataOut
  }

  def staticSelect(hiveCtx: HiveContext ): Array[String] ={
    val ignore1: Array[String] = hiveCtx.sql(s"select name from test.alading_cover_train where cover_rate<=0.2")
      .map(r => r.getAs[String](0)).collect()
    val ignore2: Array[String] = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<=1")
      .map(r => r.getAs[String](0)).collect()
    val ignore = ignore1 ++ ignore2.diff(ignore1)

    ignore2
  }

  def myLabledPoint(df: DataFrame): RDD[LabeledPoint] ={
    val featName = df.columns.drop(1)
    val dflabeledPoint: RDD[LabeledPoint] = df.map(r => LabeledPoint(r.getAs[Double](0).toInt,
      Vectors.dense(featName.map(x=>r.getAs[Double](x)))))
    dflabeledPoint
  }

  def loadData(tableName: String, hiveCtx: HiveContext): List[SchemaRDD] = {
    val train = hiveCtx.sql(s"select * from test.${tableName} where datadate<='2016-10-01'")
    val test = hiveCtx.sql(s"select * from test.${tableName} where datadate>='2017-01-01'")
    List(train,test)
  }

  def loadJsonData(fileName: String, hiveCtx: HiveContext) = {
    val jsons = hiveCtx.read.json(fileName)
    jsons.registerTempTable("testTable")
    hiveCtx.sql(s"select * from testTable")
  }

  def preProcess(df: DataFrame,hiveCtx: HiveContext): RDD[LabeledPoint] = {
    // 1.Field statistics selection
    val ignored1: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app")
    val ignored2: Array[String] = staticSelect(hiveCtx)
    val targetName: String = ("label")
    val featName: Array[String] = df.columns.diff(ignored1).diff(ignored2)
    val dfData: SchemaRDD = df.select(targetName,featName.toSeq:_*)
    println("*********** Field selection ***********")
    featName.foreach(println)
    println(featName.length)

    // 2.Numerical
    val rddFigure: RDD[LabeledPoint] = dfToDouble(dfData)
    println("*********** Convert to numeric type ***********")
    rddFigure.take(3).foreach(println)

    rddFigure
  }

  def mySampling(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val dataSample: RDD[(Int, LabeledPoint)] = data.map(row => {
      if (row.label == 1.0)
        (row, 1)
      else (row, 2)
    }).map(x => (x._2, x._1))
    val fractions: Map[Int, Double] = (List((1, 1.0), (2, 0.05))).toMap
    val approSample: RDD[(Int, LabeledPoint)] = dataSample.sampleByKey(withReplacement = false, fractions, 0)
    //approxSample.foreach(println)
    val sampDate = approSample.map(x => x._2)
    // class balance
    val aa: RDD[Int] = approSample.map(x =>x._1)
    println(aa.filter(x =>x==1).count())
    println (aa.filter(x =>x==2).count())
    sampDate
  }

  def modelBuilder(dataTrain: RDD[LabeledPoint], dataTest: RDD[LabeledPoint])={
    println("******* 4.Training *******")
    // 1.NaiveBayes -
    //val model = NaiveBayes.train(input = dataTrain, lambda = 1.0, modelType = "multinomial")

    // 2.LR - LogisticRegressionWithSGD -
    val model = LogisticRegressionWithSGD.train(input = dataTrain, numIterations = 10)

    // 3.LR - LogisticRegressionWithLBFGS -
    //val model = new LogisticRegressionWithLBFGS().setNumClasses(numClasses = 2).run(input = dataTrain)

    // 4.SVMWithSGD -
    //val model = SVMWithSGD.train(input = dataTrain, numIterations = 2)

    // 5.DecisionTree -
    //val model:DecisionTreeModel = DecisionTree.trainClassifier(input = dataTrain,numClasses = 2,
    //categoricalFeaturesInfo = collection.immutable.Map[Int,Int](),impurity = "gini", maxDepth = 5, maxBins = 32)

    // 6.RandomForest - import org.apache.spark.mllib.tree.RandomForest
    //val model = RandomForest.trainClassifier(input=dataTrain, numClasses = 2,
    //categoricalFeaturesInfo = collection.immutable.Map[Int,Int](),numTrees = 10,
    //featureSubsetStrategy = "auto", impurity = "gini", maxDepth =6, maxBins =50,seed = 123)

    // 7.GradientBoostedTrees -
    //val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    //val model = GradientBoostedTrees.train(input = dataTrain, boostingStrategy = boostingStrategy)

    // Prediction
    println("***** Begin to predict *****")
    val predictionAndLabels = dataTest.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    println ("***** Predictive - Real label (top 20) *****")
    predictionAndLabels.take(20).foreach(println)

    // ROC metrics
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val roc = metrics.roc
    val auROC = metrics.areaUnderROC
    println ("***** AUC metrics *****")
    println("Area under ROC = " + auROC)

    predictionAndLabels.collect()
  }

  def modelRF(train: RDD[LabeledPoint], test: RDD[LabeledPoint])={
    // 1.preprocessor
    //val standarScalarModel: StandardScalerModel = myStandardScalerFit(train)
    //val scalarTrain: RDD[LabeledPoint] = myStandardScalerTrans(train,standarScalarModel)
    //val scalarTest: RDD[LabeledPoint] = myStandardScalerTrans(test,standarScalarModel)
    val scalarTrain = train
    val scalarTest = test

    //val dataTrain: RDD[LabeledPoint] = mySampling(scalarTrain)
    val dataTrain = scalarTrain
    val dataTest = scalarTest

    // 6.RandomForest - import org.apache.spark.mllib.tree.RandomForest
    val model = RandomForest.trainClassifier(input=dataTrain, numClasses = 2,
      categoricalFeaturesInfo = collection.immutable.Map[Int,Int](),numTrees = 10,
      featureSubsetStrategy = "auto", impurity = "gini", maxDepth =6, maxBins =50,seed = 123)

    // 2.Prediction
    println("***** Begin to predict *****")
    val predictionAndLabels = dataTest.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    println ("***** Predictive - Real label (top 20) *****")
    predictionAndLabels.take(20).foreach(println)

    // ROC metrics
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val roc = metrics.roc
    val auROC = metrics.areaUnderROC
    println ("***** AUC metrics *****")
    println("Area under ROC = " + auROC)

    predictionAndLabels.collect()
  }

  def LRwithSGD(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): Array[(Double, Double)] ={
    // 1.preprocessor
    val standarScalarModel: StandardScalerModel = myStandardScalerFit(train)
    val scalarTrain: RDD[LabeledPoint] = myStandardScalerTrans(train,standarScalarModel)
    val scalarTest: RDD[LabeledPoint] = myStandardScalerTrans(test,standarScalarModel)

    //val dataTrain: RDD[LabeledPoint] = mySampling(scalarTrain)
    val dataTrain = scalarTrain
    val dataTest = scalarTest

    // 2.LR - LogisticRegressionWithSGD -
    val model = LogisticRegressionWithSGD.train(input = dataTrain, numIterations = 20)

    // 3.Prediction
    println("***** Begin to predict *****")
    val predictionAndLabels = dataTest.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    println ("***** Predictive - Real label (top 20) *****")
    predictionAndLabels.take(20).foreach(println)

    // 4.ROC metrics
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val roc = metrics.roc
    val auROC = metrics.areaUnderROC
    println ("***** AUC metrics *****")
    println("Area under ROC = " + auROC)

    predictionAndLabels.collect()
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("aladingModel")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("WARN")

    // 1.Load Data
    val dfOrgin = loadData("alading_train_4_10_new", hiveCtx)
    val dfTrain = dfOrgin(0)
    val dfTest = dfOrgin(1)
    println("***** 1.Table structure *****")
    dfTrain.printSchema()

    // 2.Preprocessor
    val dataTrain: RDD[LabeledPoint] = preProcess(dfTrain,hiveCtx)
    val dataTest: RDD[LabeledPoint] = preProcess(dfTest,hiveCtx)

    // 3.Sampling
    val train: RDD[LabeledPoint] = mySampling(dataTrain)
    val test = dataTest

    // 4. Build a model
    val out: Array[(Double, Double)] = LRwithSGD(train,test)
  }
}

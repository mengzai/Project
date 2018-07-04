import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType, DecimalType}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.util.random.RandomSampler
import collection._

object DXModel {
  def loadData(tableName: String, hiveCtx: HiveContext) = {
    hiveCtx.sql(s"select * from test.alading_train_4_10_new where datadate<='2016-10-01' limit 10000")
  }

  def preProcess(df: DataFrame, sqlCtx: SQLContext) = {
    import sqlCtx.implicits._
    val featuresCol: Column = array(df.columns.filter(x => x != "label" && x != "ecif_id" && x != "datadate" && x != "ecif_id_activity"
      &&x != "datadate_activity" && x != "ecif_id_communicate" && x != "datadate_communicate" && x != "ecif_id_kyc" && x != "datadate_kyc"
      && x != "ecif_id_order" && x != "datadate_order" && x != "ecif_id_trade" && x != "datadate_trade" && x != "datadate_app"
      && x != "datadate_web" && x != "u_id_app"  && x != "u_id_web" && x != "num"  && x != "num_app").map(col).map(_.cast(DoubleType)): _*)
    df.select(col("label").cast(DoubleType), featuresCol)
      .map(r => LabeledPoint(r.getAs[Double](0), Vectors.dense(r.getAs[mutable.WrappedArray[Double]](1).toArray)))
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DXModel")
    val sc = new SparkContext(conf)
    //Load Data
    val hiveCtx = new HiveContext(sc)
    val trInput = loadData("dianxiao_train", hiveCtx)
    trInput.printSchema()

    //Pre-Process
    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._
    val train = preProcess(trInput, sqlCtx)
    train.take(20).foreach(println)
    println(train.count())

    /* 
    //Sampling
    val fractions: Map[Double, Double] = Map(1.0 -> 1.0, 0.0 -> 0.02)
    //val sampledTrain = train.sampleByKeyExact(withReplacement = false, fractions, 1)
    val sampledTrain = train.filter{ case LabeledPoint(label, features) => 
      label == 1 && 

    println(sampledTrain.count())
    //Training
    //LR
    val lr = new LogisticRegressionWithSGD()
    lr.optimizer
      .setNumIterations(10)
      .setUpdater(new L1Updater)
      .setRegParam(0.01)

    println("Start Training")
    val model = lr.run(train)
    */

    //RF
    val model = RandomForest.trainClassifier(train, 2, collection.immutable.Map[Int, Int](), 10, "auto", "gini", 6, 100, 123)

    val teInput = loadData("dianxiao_test", hiveCtx)
    val test = preProcess(teInput, sqlCtx)
    test.take(20).foreach(println)
    val predictionAndLabels = test.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    predictionAndLabels.take(20).foreach(println)
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    // ROC Curve
    val roc = metrics.roc
    // AUROC
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)
  }
}

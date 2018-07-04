/**
  * Created by wenqiang on 2017/7/21.
  */
import java.io

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext, ml}

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
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier, LogisticRegression, LogisticRegressionModel, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.mllib.linalg

import scala.runtime.Nothing$

object AladingModel3 {
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

  def fieldDistinctCount(data: DataFrame,processingNumber:Int = 100): Array[(String, Long)] = {
    val fieldName: Array[String] = data.columns
    var processingField: Array[String] = fieldName
    var distinctCountOut: List[Long] = List()
    //  Iterative processing
    while (processingField.length>processingNumber){
      val subField: Array[String] = processingField.slice(0,processingNumber)
      val distinctCount: Seq[Long] =data.agg(countDistinct(subField(0)),subField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
        .first().toSeq.asInstanceOf[Seq[Long]]
      distinctCountOut = distinctCountOut ++ distinctCount
      processingField = processingField.diff(subField)
      println(distinctCount)
    }
    distinctCountOut = distinctCountOut ++ data.agg(countDistinct(processingField(0)),processingField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
      .first().toSeq.asInstanceOf[Seq[Long]]
    fieldName.zip(distinctCountOut)
  }

  def fieldCoverRate(data: DataFrame): Array[(String, Double)] = {
    val fieldName: Array[String] = data.columns
    val coverNum = data.agg(count(fieldName(0)),fieldName.drop(1).map(x=>count(x)).toSeq:_*)
    val size: Long = data.count()
    val coverRateValue: Seq[Double] = coverNum.first().toSeq.asInstanceOf[Seq[Long]].map(x=>x*1.0/size)
    fieldName.zip(coverRateValue)
  }

  def Sampling(hiveCtx: HiveContext,data: DataFrame,positiveProbability: Double,negativeProbability: Double,
               labelName: String = "label"): SchemaRDD = {
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

    val dfName = data.columns
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfSample = hiveCtx.createDataFrame(generatedDate, schema)
    dfSample
  }

  def featureImportancesDump (sc:SparkContext,hiveCtx:HiveContext,model: RandomForestClassificationModel,
                              featureFields: Array[String],tableName: String) = {
    val featureImData: Array[(Double, String)] = model.featureImportances.toArray.zip(featureFields).sortBy(_._1)
    val featureImRdd: RDD[(Double, String)] = sc.parallelize(featureImData)
    val featureImDf = hiveCtx.createDataFrame(featureImRdd).toDF("name","featureImportance")
    injectIntoHive(hiveCtx,featureImDf,tableName)
  }

  def injectIntoHive(hiveCtx: HiveContext,data: DataFrame,tableName: String,dbName:String = "test") = {
    data.registerTempTable("tempTable")
    hiveCtx.sql(s"use $dbName")
    hiveCtx.sql(s"create table $tableName as select * from tempTable")
  }

  def preProcess(hiveCtx: HiveContext,df: DataFrame,labelField: String,featuresFields:Array[String],
                 indexSample: Int = 0): SchemaRDD = {
    // 1.numerical processing
    var selectData: SchemaRDD = numericalProcessing(df.select(labelField,featuresFields.toVector.toSeq:_*),hiveCtx)
    // 2.sample
    if (indexSample == 1){
      selectData = Sampling(hiveCtx,selectData,positiveProbability = 1.0 ,negativeProbability=0.045)
    }
    selectData
  }

  def RFModel(trainData:DataFrame,testData:DataFrame): (PipelineModel, SchemaRDD) ={
    val dataTrainData = trainData
    val dataTestData = testData

    /* 1. model building  */
    val fields = dataTrainData.columns
    val labelField = fields(0)
    val featureFields: Array[String] = fields.drop(1)

    // Stage 1
    val labelIndexer: StringIndexerModel = new StringIndexer()
      .setInputCol(labelField)
      .setOutputCol("label_idx")
      .fit(dataTrainData)

    // Stage 2
    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureFields)
      .setOutputCol("featureVector")

    // Stage 3
    val rfClassifier: RandomForestClassifier = new RandomForestClassifier()
      .setLabelCol("label_idx")
      .setFeaturesCol("featureVector")
      .setMaxBins(100).setMaxDepth(12).setNumTrees(100).setImpurity("gini")

    // Stage 4
    val labelConverter: IndexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictionLabel")
      .setLabels(labelIndexer.labels)

    // Stage 5
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(labelIndexer, vectorAssembler,rfClassifier,labelConverter))

    val model: PipelineModel = pipeline.fit(dataTrainData)

    /* 2.Make predictions */
    val predictions = model.transform(dataTestData)
    predictions.select(labelField,"label_idx","probability","prediction","predictionLabel").show(50)

    /* 3.ROC metrics  */
    val predictScores = predictions.select("probability", "label_idx").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics = new BinaryClassificationMetrics(predictScores)
    val auROC = metrics.areaUnderROC
    println ("***** AUC metrics *****")
    println("Area under ROC = " + auROC)

    (model,predictions)
  }

  def RFModel(data:DataFrame): (PipelineModel, SchemaRDD) = {
    val Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2))
    RFModel(trainData,testData)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("aladingModel")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("WARN")

    /*  1.Load Data  */
    val rawTrainData = hiveCtx.sql(s"select * from test.alading_alldata_end_5 where datadate<='2016-10-01'")
    val rawTestData = hiveCtx.sql(s"select * from test.alading_alldata_end_5 where datadate>='2017-04-01'")

    /*  2.Preprocessor  */
    val labelField: String = s"label"
    val invalidFields: List[String] = List("ecif_id","datadate","ecif_id_trade","datadate_trade",
      "ecif_id_activity","ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc","ecif_id_order",
      "datadate_order","u_id_app","datadate_app","u_id_web","datadate_web","num","num_app","recent_order_product",
      "recent_order_days","recent_investment_days","recent_investment_due_days")
    //val lowerCoverFields: Array[String] = fieldCoverRate(rawTrainData).filter(x=>x._2<=0.2).map(x=>x._1)
    val singleValueFields: Array[String] = fieldDistinctCount(rawTrainData).filter(x =>x._2<=1).map(x=>x._1)
    val featureFields: Array[String] = rawTrainData.columns
      .diff(List(labelField)).diff(invalidFields).diff(singleValueFields)//.diff(lowerCoverFields)
    val dataTrainData = preProcess(hiveCtx,rawTrainData,labelField,featureFields,indexSample = 1)
    val dataTestData = preProcess(hiveCtx,rawTestData,labelField,featureFields)

    /* 3.ModelBuilder */
    val modelOutput = RFModel(dataTrainData,dataTestData)
    val pipelineModel: PipelineModel = modelOutput._1
    val metricsResult: SchemaRDD = modelOutput._2

    /* 4.Save - model */
    val path = s""
    pipelineModel.save(path)
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /* 5.save - metricsResult */
    val metricsData = metricsResult.select(labelField,"probability","predictionLabel")
      .map(r => (r(0).toString,r(1).toString.split(",")(0).replace("[",""),r(2).toString))
    val ecifidData = rawTestData.select("ecif_id","datadate").map(x=>(x(0).toString,x(1).toString))
    val outputData: RDD[(String, String, String, String, String)] =
      ecifidData.zip(metricsData).map(x=>(x._1._1,x._1._2,x._2._1,x._2.x._2,x._2._3))
    val metricsDf = hiveCtx.createDataFrame(outputData).toDF("ecif_id","datadate","label","prediction","probability")
    injectIntoHive(hiveCtx,metricsDf,tableName = "alading_metrics_test")

    /* 6.Save - distinctCount| CoverRate */
    val distinctCount: RDD[(String, Long)] = sc.parallelize(fieldDistinctCount(rawTrainData).map(x=> (x._1,x._2)))
    val distinctCountDf = hiveCtx.createDataFrame(distinctCount).toDF("field_name","countDistinct")
    injectIntoHive(hiveCtx,distinctCountDf,tableName = "alading_distinctCount_test")

    val coverRate: RDD[(String, Double)] = sc.parallelize(fieldCoverRate(rawTrainData).map(x=>(x._1,x._2)))
    val coverRateDf = hiveCtx.createDataFrame(coverRate).toDF("field_name","coverRate")
    injectIntoHive(hiveCtx,coverRateDf,tableName = "alading_coverRateDf_test")
  }
}





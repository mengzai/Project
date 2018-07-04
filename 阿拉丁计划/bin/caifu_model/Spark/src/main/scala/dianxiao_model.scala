package cn.creditease.invest.test
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
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.mllib.linalg

import scala.runtime.Nothing$

object dianxiao_model {

  def load_data(hiveCtx:HiveContext)={
    val good_df = hiveCtx.sql(s"select * from test.dainxiao_model_data where label=1")
    val bad_df = hiveCtx.sql(s"select * from test.dainxiao_model_data where label=0")

    val valid_data = hiveCtx.sql(s"select * from test.dainxiao_validate ")
    println("好人 数量：",good_df.count())
    println("坏人 数量：",bad_df.count())

    val good_data = good_df.randomSplit(Array(0.7, 0.3), seed = 25L)
    val train_good: SchemaRDD = good_data(0).cache()
    val test_good = good_data(1)

    val bad_data = bad_df.randomSplit(Array(0.7, 0.3), seed = 25L)
    val train_bad = bad_data(0).cache()
    val test_bad = bad_data(1)


    val train_df=train_good.unionAll(train_bad)
    val test_df=test_good.unionAll(test_bad)

    (train_df,test_df,valid_data)
  }

  def fieldCoverRate(data: DataFrame): Array[(String, Double)] = {
    val fieldName: Array[String] = data.columns
    val coverNum = data.agg(count(fieldName(0)),fieldName.drop(1).map(x=>count(x)).toSeq:_*)
    val size: Long = data.count()
    val coverRateValue: Seq[Double] = coverNum.first().toSeq.asInstanceOf[Seq[Long]].map(x=>x*1.0/size)
    fieldName.zip(coverRateValue)
  }

  def fieldDistinctCount(data: DataFrame,processingNumber:Int = 100): Array[(String, Long)] = {
    val fieldName: Array[String] = data.columns     //取数据的列名
    var processingField: Array[String] = fieldName  //作为中间变量
    var distinctCountOut: List[Long] = List()
    //  Iterative processing
    while (processingField.length>processingNumber){
      val subField: Array[String] = processingField.slice(0,processingNumber) //每次取100个进行批量跑批覆盖率
      val distinctCount: Seq[Long] =data.agg(countDistinct(subField(0)),subField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
        .first().toSeq.asInstanceOf[Seq[Long]]
      distinctCountOut = distinctCountOut ++ distinctCount //
      processingField = processingField.diff(subField)
      println(distinctCount)

    }

    distinctCountOut = distinctCountOut ++ data.agg(countDistinct(processingField(0)),processingField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
      .first().toSeq.asInstanceOf[Seq[Long]]
    fieldName.zip(distinctCountOut)
  }

  def Select_feature(train_df:DataFrame) ={
    val labelField: String = s"label"
    val invalidFields: List[String] = List(
      "gender",
      "id_type",
      "apply_times",
      "loan_count",
      "query_times",
      "phone",
      "id_number",
      "education",
      "validate_date",
      "live_case",
      "relation",
      "org_type",
      "org_city",
      "entry_date",
      "company_position",
      "house_type",
      "year_income",
      "support_persons",
      "has_child",
      "child_nums",
      "id_validity_date",
      "resident_province",
      "recruitment_date",
      "house",
      "house_city_real",
      "monthly_outlay",
      "house_condition",
      "in_city_years",
      "borrower_type",
      "customer_ident_type",
      "share_proportion",
      "corp_regist_capital",
      "is_abnormal",
      "month_income",
      "risk_industry",
      "product_type",
      "loan_type",
      "money_source",
      "grade_version",
      "credit_level",
      "wjq_count",
      "system_source_name"
    )
    //val lowerCoverFields: Array[String] = fieldCoverRate(rawTrainData).filter(x=>x._2<=0.2).map(x=>x._1)
    val singleValueFields: Array[String] = fieldDistinctCount(train_df).filter(x =>x._2<=1).map(x=>x._1)
    val featureFields: Array[String] = train_df.columns
      .diff(List(labelField)).diff(invalidFields).diff(singleValueFields)//.diff(lowerCoverFields)
    (labelField,featureFields)
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

  def numericalProcessing(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
    } catch {
      case ex: NumberFormatException => -999.0
    }
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def dataProcessing(df: DataFrame, hiveCtx:HiveContext): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>numericalProcessing(r.getAs(x))).toSeq:_*))
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = hiveCtx.createDataFrame(data, schema)
    dfFigure
  }

  def preProcess(hiveCtx: HiveContext,df: DataFrame,labelField: String,featuresFields:Array[String],
                 indexSample: Int = 0): SchemaRDD = {
    // 1.numerical processing  数据格式处理
    var selectData: SchemaRDD = dataProcessing(df.select(labelField,featuresFields.toVector.toSeq:_*),hiveCtx)
    // 2.sample
    if (indexSample == 1){
      selectData = Sampling(hiveCtx,selectData,positiveProbability = 1.0 ,negativeProbability=1.0)
    }
    selectData
  }

  def preProcess_vaild(hiveCtx: HiveContext,df: DataFrame,labelField: String,featuresFields:Array[String],
                 indexSample: Int = 0): SchemaRDD = {
    // 1.numerical processing  数据格式处理
    var selectData: SchemaRDD = dataProcessing(df.select(featuresFields.map(col):_*),hiveCtx)
    // 2.sample
    if (indexSample == 1){
      selectData = Sampling(hiveCtx,selectData,positiveProbability = 1.0 ,negativeProbability=1.0)
    }
    selectData
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
    hiveCtx.sql("use test")
    hiveCtx.sql("drop table dainxiao_Feature_important_new")
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql("create table dainxiao_Feature_important_new as select * from tmp_table")
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

//  def RFModel(dataTrainData:DataFrame,dataTestData:DataFrame, label: String, featuresFields:Array[String],sc:SparkContext,hiveCtx:HiveContext,features: String = "features"): SchemaRDD ={
//
//    val indexer = new StringIndexer()
//      .setInputCol(label)
//      .setOutputCol("label_idx")
//      .fit(dataTrainData)
//
//    val featureIndexer: VectorIndexerModel = new VectorIndexer()
//      .setInputCol(features)
//      .setOutputCol("features_idx")
//      .fit(dataTrainData)
//
//    val RF = new RandomForestClassifier().setLabelCol("label_idx").setMaxBins(32).
//      setMaxDepth(12).setNumTrees(100).setMinInstancesPerNode(8).setImpurity("gini")
//    //val GBT = new GBTClassifier().setLabelCol("label_idx").setMaxBins(32)
//    //    .setMaxDepth(6).setMinInstancesPerNode(4).setImpurity("gini").setMaxIter(10)
//
//    // Train model. This al so runs the indexers.
//
//    val model: RandomForestClassificationModel = RF.fit(featureIndexer.transform(indexer.transform(dataTrainData)))
//
//    Feature_important(model,featuresFields,sc,hiveCtx)
//
//    //val model: GBTClassificationModel = GBT.fit(featureIndexer.transform(indexer.transform(dataTrainData)))
//    // Make predictions.
//    val predictions_train = model.transform(dataTrainData)
//
//    val converter = new IndexToString()
//      .setInputCol("probabilication")
//      .setOutputCol("label_new")
//      .setLabels(indexer.labels)
//
//    // ROC metrics
//    val predictScores_train = predictions_train.select("probability", "label_new").rdd
//      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
//    val metrics_train = new BinaryClassificationMetrics(predictScores_train)
//    val auROC_train = metrics_train.areaUnderROC
//    println("RF 训练集  AUC" + auROC_train)
//
//    //测试集
//    val predictions = model.transform(dataTestData)
//    predictions_train.select(label,"label_idx","probability","prediction","predictionLabel").show(50)
//    val predictScores = predictions.select("probability", "label").rdd
//      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
//    val metrics = new BinaryClassificationMetrics(predictScores)
//    val auROC = metrics.areaUnderROC
//    println("RF 测试集  AUC" + auROC)
//    predictions
//  }

  def RFModel(dataTrainData:DataFrame,dataTestData:DataFrame,datavalidData:DataFrame, label: String, featuresFields:Array[String],sc:SparkContext,hiveCtx:HiveContext,features: String = "features") ={

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
      .setMaxBins(50).setMaxDepth(6).setNumTrees(50).setImpurity("gini")
    val rfmodel=rfClassifier.fit(vectorAssembler.transform(labelIndexer.transform(dataTrainData)))

//    Feature_important(rfmodel,featureFields,sc,hiveCtx)

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

    /* 3.ROC metrics  */
    println("train model AUC \n")
    val predictions_train = model.transform(dataTrainData)
//    predictions_train.select(labelField,"label_idx","probability","prediction","predictionLabel").show(50)

    val predictScores_train = predictions_train.select("probability", "label_idx").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics_train = new BinaryClassificationMetrics(predictScores_train)
    val auROC_train = metrics_train.areaUnderROC
    println("Area under ROC = " + auROC_train)

    println("test model  AUC \n")
    val predictions = model.transform(dataTestData)
//    predictions.select(labelField,"label_idx","probability","prediction","predictionLabel").show(50)

    val predictScores = predictions.select("probability", "label_idx").rdd
      .map(r => (r(0).toString.split(",")(1).replace("]","").toDouble, r(1).toString.toDouble))
    val metrics = new BinaryClassificationMetrics(predictScores)
    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)



    println("valid model  AUC \n")
    val predictions_valid = model.transform(datavalidData)
    (model,predictions,predictions_valid)

  }

  def injectIntoHive(hiveCtx: HiveContext,data: DataFrame,tableName: String,dbName:String = "test") = {

    data.registerTempTable("tempTable")
    hiveCtx.sql(s"use $dbName")
    hiveCtx.sql(s"create table $tableName as select * from tempTable")
  }

  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setAppName("final_alading_model")
    val sc=new SparkContext(conf)  //创建第一个RDD
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")//不打印info

    println("### load data ###")
    val (train_df,test_df,valid_data)=load_data(hiveCtx)

    println("训练集 数量：",train_df.count())
    println("测试集 数量：",test_df.count())
    println("验证集 数量：",valid_data.count())

    println("###  select feature ###")
    val  (labelField,featureFields)=Select_feature(train_df)
    val dataTrainData = preProcess(hiveCtx,train_df,labelField,featureFields,indexSample = 1)
    val dataTestData = preProcess(hiveCtx,test_df,labelField,featureFields)
    val datavalidData = preProcess_vaild(hiveCtx,valid_data,labelField,featureFields)


    /* 3.ModelBuilder */
    println("###  model  ###")
    val modelOutput = RFModel(dataTrainData,dataTestData,datavalidData,labelField,featureFields,sc,hiveCtx)
    val pipelineModel: PipelineModel = modelOutput._1
    val metricsResult: SchemaRDD = modelOutput._2
    val datavalidresult: SchemaRDD = modelOutput._3

//    datavalidresult.select("id_number","probability","predictionLabel").show()

//    //    /* 4.Save - model */
//    //    val path = s""
//    //    pipelineModel.save(path)
//    //    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    //
//    //    /* 5.save - metricsResult */

//    val metricsData = metricsResult.select(labelField,"probability","predictionLabel")
//      .map(r => (r(0).toString,r(1).toString.split(",")(0).replace("[",""),r(2).toString))
//    val metricsDf = hiveCtx.createDataFrame(metricsData).toDF("label","prediction","probability")
//  injectIntoHive(hiveCtx,metricsDf,tableName = "dianxiao_metrics_test")

    //into_valid_data_todatabase

    val Id_number = valid_data.select("id_number").map(x=>x.toString.replace("[","").replace("]",""))

    val metricsvalid = datavalidresult.select("probability","predictionLabel")
      .map(r => (r(0).toString.split(",")(1).replace("]",""),r(1).toString))

    val outputData: RDD[(String, String, String)] = Id_number.zip(metricsvalid).map(x=>(x._1,x._2._1,x._2.x._2))

    val metrics_val = hiveCtx.createDataFrame(outputData).toDF("id_number","probability","predictionLabel")

    metrics_val.show()
    injectIntoHive(hiveCtx,metrics_val,tableName = "dianxiao_valid")
    
    /* 6.Save - distinctCount| CoverRate */
    //    val distinctCount: RDD[(String, Long)] = sc.parallelize(fieldDistinctCount(test_df).map(x=> (x._1,x._2)))
    //    val distinctCountDf = hiveCtx.createDataFrame(distinctCount).toDF("field_name","countDistinct")
    //    injectIntoHive(hiveCtx,distinctCountDf,tableName = "alading_distinctCount_test")
    //
    //    val coverRate: RDD[(String, Double)] = sc.parallelize(fieldCoverRate(test_df).map(x=>(x._1,x._2)))
    //    val coverRateDf = hiveCtx.createDataFrame(coverRate).toDF("field_name","coverRate")
    //        injectIntoHive(hiveCtx,coverRateDf,tableName = "alading_coverRateDf_test")
  }
}





package cn.creditease.invest.test

import java.text.SimpleDateFormat

import cn.creditease.invest.test.line_down_spark_test.inforEntropy
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.feature.StandardScalerModel
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
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.util.random.RandomSampler
import org.apache.spark.mllib.feature.ChiSqSelector
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

import scala.collection.mutable.ListBuffer
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{Bucketizer, VectorAssembler}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DecimalType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel, TreeEnsembleModel}
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
//import org.apache.spark.mllib.util.{MLUtils, Saveable}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.util.random.RandomSampler
import org.apache.spark.mllib.feature.{ChiSqSelector, ChiSqSelectorModel, Normalizer, StandardScaler}

import collection.{Seq, _}
import org.apache.spark.mllib.stat.Statistics
import org.json4s.jackson.Json
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vectors
import java.util.Date
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}


object SparkTest {

  def myToDouble(x: Any) = x match{
    case s: String => try{s.toDouble}catch{case ex: NumberFormatException => -999.0}
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def getNowDate():String={
    var now:Date = new Date()
    var dateFormat:SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    var now_time = dateFormat.format( now )
    now_time
  }

  //覆盖率优化
  def  Feature_coverage_optimize(df:DataFrame,sc:SparkContext,hiveCtx: HiveContext)={
    val all_length=df.count().toDouble
    val columnName = df.columns

    println("all colmons describe")
    print("describe开始时间为：",getNowDate())
    val dfDescrIn = df.describe(columnName.toSeq:_*)
    print("describe结束时间为：",getNowDate())

    // save DataFrame test
    val fieldName: Array[String] = dfDescrIn.columns.drop(1)
    val dataTranspose: Array[Array[String]] = fieldName.map(c =>
      (Array(c)++(dfDescrIn.select(c).map(r => r.getAs[String](0)).collect())))

    // Trans to DataFrame
    val dfName: Array[String] = Array("field_name")++(dfDescrIn.map(x =>x.getAs[String](0)).collect())

    val data: RDD[Row] = sc.parallelize(dataTranspose.map(r => Row(r.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescrOut = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/all_length)
    dfDescrOut.show()

    // insert into hive
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_describe_cover_rate_test as select * from tmp_table")
  }

  def Can_ToDouble(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
      1
    } catch {
      case ex: NumberFormatException => 0
    }
    case jn: java.lang.Number => 1
    case _ => 0
  }
  //覆盖率优化
  def  Feature_coverage_optimize_N(df:DataFrame,sc:SparkContext,hiveCtx:HiveContext)={
    val all_length=df.count().toInt
    val dfName: Array[String] = df.columns
    var key_val: mutable.Map[Int, Double] = scala.collection.mutable.Map()
    val columnName = df.columns
    println("###########开始 求解覆盖率  且开始时间为:",getNowDate())

    df.collect().foreach(
      x=> for (i <-1 to x.length){
        println(x)
        if(key_val.contains(i)) key_val(i)=key_val(i)+Can_ToDouble(x.get(i-1))
        else key_val(i)=Can_ToDouble(x.get(i-1))
      }
    )

    println("###########开始 求解覆盖率  且结束时间为:",getNowDate())
    val new_map=key_val.map(x=>(dfName(x._1-1).toString,(x._2/all_length).toString))
    val data: RDD[Row] = sc.parallelize(new_map.map(x =>Row(x._1,x._2)).toVector)

    val target_name: Array[String] = Array("field_name","cover_rate")
    val schema = StructType(target_name.map(fieldName => StructField(fieldName, StringType, true)))
    val Df_cover_rate = hiveCtx.createDataFrame(data, schema)
    Df_cover_rate.show()
    println("###########放入表中#########")

    Df_cover_rate.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_cover_rate_test as select * from tmp_table")

  }

  def fieldCoverRate(data: DataFrame): Array[(String, Double)] = {
    val fieldName: Array[String] = data.columns
    val coverNum = data.agg(count(fieldName(0)),fieldName.drop(1).map(x=>count(x)).toSeq:_*)
    println("***************** Coverage Rate running ****************")
    val size: Long = data.count()
    val coverRateValue: Seq[Double] = coverNum.first().toSeq.asInstanceOf[Seq[Long]].map(x=>x*1.0/size)
    coverRateValue.foreach(println)
    fieldName.zip(coverRateValue)
  }

  def Feature_coverage(df:DataFrame,sc:SparkContext)= {
    val all_length=df.count()
    val buf = new ListBuffer[Any]
    for (x: String <- df.columns) {
      val ds =df.describe(x)//des结果
      val result: RDD[Any] = ds.toDF().rdd.map(it => {it.get(it.fieldIndex(x))})
      val t: Array[Any] = result.collect()
      val Cnt: Int =ds.first().get(1).toString.toInt
      val cover=Cnt*1.0/all_length

      val site =(x,Cnt,cover,t.toVector)
      buf.append(site)
    }

    //    存储文件
    sc.parallelize(buf.toVector)
      .repartition(1).saveAsTextFile("/user/hive/warehouse/db/test.db/alading_cover_trian_1")
  }

  //应用信息增益选择特征
  def Use_inforGain_find_features(sc: SparkContext,df:DataFrame,topK:Int,ignore_name:Seq[String])={
    println("####################start   start######################")
    val myMap = new scala.collection.mutable.HashMap[String, Double]
    val buf = new ListBuffer[Any]
    for (x: String <- df.columns.diff(ignore_name)) {
      val featureInd: Int = df.columns.indexOf(x)
      val targetInd: Int = df.columns.indexOf("label")
      val label_feature_RDD: RDD[(Double, Double)] =df.map(r => (myToDouble(r.getAs(targetInd)),myToDouble(r.getAs(featureInd))))
      val Inforgainrate =inforGainRate(sc,label_feature_RDD)
      println(x,Inforgainrate)
      buf.append(x,Inforgainrate)
      myMap += (x -> Inforgainrate)
    }

    sc.parallelize(buf.toVector)
      .repartition(1).saveAsTextFile("/user/hive/warehouse/db/test.db/alading_info_value_all")
//    val mapSortSmall = myMap.toList.sortBy(_._2)//从小到大
    val mapSortBig = myMap.toList.sortBy(-_._2)   //从大到小

//    返回前K个列名
    val Topk_vec: List[String] =mapSortBig.take(topK).map(x=>x._1)
    println("####################从大到小打印特征重要性   start######################")
    Topk_vec.foreach(println)
    println("####################从大到小打印特征重要性  end#########################")
    Topk_vec
  }

  def feature_extract(df: DataFrame,sc:SparkContext) = {
    // index of target
    val targetInd: Int = df.columns.indexOf("label")

    // index of feat (exclude columns)
    val ignored: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app")

    val ignored_and_label: Seq[String] = List("ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app")

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


  //CV
//  def CrossValidator(trainingData:RDD[LabeledPoint],testData:RDD[LabeledPoint])={
//
//    val nFolds: Int = 10
//    val NumTrees: Int = 3
//    val indexer = new StringIndexer().setInputCol("label").setOutputCol("label_idx")
//    val rf = new RandomForestClassifier().setNumTrees(NumTrees).setFeaturesCol("features").setLabelCol("label_idx")
//    val pipeline = new Pipeline().setStages(Array(indexer, rf))
//    val paramGrid = new ParamGridBuilder().build()
//    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
//    val cv = new CrossValidator() .setEstimator(pipeline) .setEvaluator(evaluator) .setEstimatorParamMaps(paramGrid) .setNumFolds(nFolds)
//    val model = cv.fit(trainingData)
//    val predictions = model.transform(testData)
//    // Show model predictions
//    predictions.show()
//    val accuracy = evaluator.evaluate(predictions)
//    println("Accuracy: " + accuracy)
//    println("Error Rate: " + (1.0 - accuracy))
//
//  }
  //get_train_test

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



  def  GBDT_model_ml(sqlContext: SQLContext)={


    val data: SchemaRDD = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model.  This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
    println("Learned regression GBT model:\n" + gbtModel.toDebugString)
  }

  //
  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  //信息熵
  def inforEntropy(sc:SparkContext,target_attribute: Array[Double]): Double = {
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
    val target: Array[Double] = feature_attribute.map { x => x._2 }.toArray()
    val Entropy1 = inforEntropy(sc,target)
    val all_Entropy = sc.accumulator(0.0)
    feature_attribute.groupBy(x => x._1).foreach { x => all_Entropy += (x._2.size.toDouble / target.length) * inforEntropy(sc,x._2.map(x => x._2).toArray)}
    val X = feature_attribute.map { x => x._1 }
    val Y = feature_attribute.map { x => x._2 }
    val correlation: Double = Statistics.corr(X, Y, "pearson")
//    println(Entropy1)
//    println(all_Entropy.value)
    ((Entropy1 - all_Entropy.value), correlation)
  }

  //计算特征与目标特征之间的信息增益率
  def inforGainRate(sc: SparkContext, feature_attribute: RDD[(Double, Double)]) = {
    val target = feature_attribute.map { x => x._2 }.toArray()
    val Entropy1 = inforEntropy(sc,target)
    val all_Entropy = sc.accumulator(0.0)
    feature_attribute.groupBy(x => x._1).collect().foreach(x=>all_Entropy+=(x._2.size.toDouble*1.0/target.length)*inforEntropy(sc,x._2.map(x=>x._2).toArray))
    val X = feature_attribute.map { x => x._1 }
    val Y = feature_attribute.map { x => x._2 }
    val correlation: Double = Statistics.corr(X, Y, "pearson")
    /*    // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
72     // If a method is not specified, Pearson's method will be used by default.
73     val correlMatrix: Matrix = Statistics.corr(data, "pearson")*/
//    println(all_Entropy.value)
    (Entropy1 - all_Entropy.value).toDouble/inforEntropy(sc,X.toArray())
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

    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
    import hiveContext.implicits._
    hiveContext.sql("use test")
//    val Path =
    val rawblocks = sc.textFile("/user/hive/warehouse/db/test.db/alading_cover_trian_1_copy/part-00000").map(x => Parse(x))
    rawblocks.toDF()insertInto("alading_cover_train")

//    hiveContext.sql("insert into alading_cover_test_1 " +
//      "select name, cnt,cover_rate,mean,min,max,std from table1")
    println("################end###################")
  }

  def mySampling(data: RDD[LabeledPoint],sand:Long=11l): RDD[LabeledPoint] = {

    val dataSample: RDD[(Int, LabeledPoint)] = data.map(row => {
      if (row.label == 1.0)
        (row, 1)
      else (row, 2)
    }).map(x => (x._2, x._1))
    val fractions: Map[Int, Double] = (List((1, 1.0), (2, 0.04))).toMap
    val approSample = dataSample.sampleByKey(withReplacement = false, fractions, 0)
    //approxSample.foreach(println)
    approSample.map(x => x._2)
  }

  def Get_features_and_todouble(df: DataFrame,sc:SparkContext,name:Seq[String])={
    //根据卡方得到的name 选择特征
    val data: RDD[Row] = df.map(r =>Row(name.map(x =>myToDouble(r.getAs(x))).toSeq:_*))
    val sqlContext = new SQLContext(sc)
    val schema = StructType(name.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = sqlContext.createDataFrame(data, schema)

//    // index of target
    val targetInd: Int = dfFigure.columns.indexOf("label")
    // index of feat (exclude columns)
    val ignored: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app","Recent_order_days","Recent_investment_days","Recent_investment_due_days")

    val featInd: Array[Int] = dfFigure.columns.diff(ignored).map(dfFigure.columns.indexOf(_))
    val data_get=dfFigure.map(r => LabeledPoint(myToDouble(r.getAs(targetInd)), Vectors.dense(featInd.map(x=>myToDouble(r.getAs(x))))))
   (dfFigure,data_get)

  }

  def get_Disperse_ChiSqSel(df: DataFrame,hiveCtx:HiveContext,dispersed_index:Seq[String]) = {

    println("###########statr###########")
    val dfName: Array[String] = df.columns
    val label: String = dfName(0)
    val Feature_out_label: Array[String] = dfName.drop(1)

    val Feature_continuous: Array[String] =  Feature_out_label.diff(dispersed_index)  //找到连续的特征
    val featOut: Array[String] = (dispersed_index.toArray)++Feature_continuous        //离散值+连续值

    //设置管道
    println("###########离散化###########")
    import org.apache.spark.ml.feature.QuantileDiscretizer
    import org.apache.spark.ml.Pipeline
    val discretizers = Feature_continuous.map(c =>
      new QuantileDiscretizer().setInputCol(c).setOutputCol(s"${c}_disc").setNumBuckets(100))
    val pipeline: Pipeline = new Pipeline().setStages(discretizers)
    val dfDisc = pipeline.fit(df).transform(df)

    // 卡方选择
    println("###########卡方选择###########")
    val featChi: Array[String] = dfDisc.columns.diff((label++Feature_continuous).toSeq)  //离散值
    val dataChi: RDD[LabeledPoint] = dfDisc.map(r => LabeledPoint(r.getAs[Double](label).toInt,
      Vectors.dense(featChi.map(x => r.getAs[Double](x)))))
    val selector = new ChiSqSelector(50)
    val transformer = selector.fit(dataChi)
    println("The selectedFeatures's index")
    println(transformer.selectedFeatures.mkString(" ~ "))
    val selectName: Array[String] = transformer.selectedFeatures.map(x => featOut(x))
    println("The selectedFeatures's Name")
    println(selectName.mkString(" ~ "))

//    val filteredData=dataChi.map{lp =>LabeledPoint(lp.label,transformer.transform(lp.features))}

    val lab = Array("label")
    println("###########end###########")
    lab++selectName

  }



//  特征的相关性处理：
  def get_feature_Select(train_df:DataFrame,test_df:DataFrame,sc:SparkContext,name:Seq[String],HiveCtx:HiveContext) ={
    //全集
    val (targer_train_df ,data_train_get)=feature_extract(train_df,sc)
    val (targer__test_df ,data_test_get)=feature_extract(test_df,sc)
    //覆盖率选择特征
    fieldCoverRate(targer_train_df)
//    Feature_coverage_optimize_N(targer_train_df,sc,HiveCtx)//保存到/user/hive/warehouse/db/test.db/alading_cover_train/
    //对覆盖率进行提取特征s
//
//    // 信息增益选择特征 val Topk_vec: List[String]
//    val HiveCtx = new HiveContext(sc)
//    val ignoreDisc: Seq[String] = HiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<20")
//      .map(r => r.getAs[String](0)).collect().toSeq
//    val Topk_vec: List[String]=Use_inforGain_find_features(sc,targer_train_df,10,ignoreDisc)
//
//    //将特征覆盖率及简单describe 在hive层建表
//        get_the_list_to_excle(sc)
//
//    //特征选择+处理
//    //根据覆盖率或者信息增益结果选择最终数据集合
//    val (targer_train_df: SchemaRDD,data_train_get: RDD[LabeledPoint])=Get_features_and_todouble(train_df,sc,name)
//    val (targer_test_df,data_test_get)=Get_features_and_todouble(test_df,sc,name)
//    val Topk_vec=Use_inforGain_find_features(sc,targer_train_df,100,ignoreDisc)
  }


  def dfToDouble(df: DataFrame, hiveCtx:HiveContext): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>myToDouble(r.getAs(x))).toSeq:_*))
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

    mlData
  }

  def out_pre(hiveCtx:HiveContext,preLabel:RDD[(Double, Double)])={
    // 4. Build a model

    // 5.The result of running by day

    val ecifTime: RDD[(String, String)] =
      hiveCtx.sql(s"select ecif_id,datadate from test.alading_train_4_10_new where datadate>='2017-01-01'")
        .map(r=>(r.getAs[String](0),r.getAs[String](1)))
    val preData: RDD[Row] = ecifTime.zip(preLabel).map(r=>Row(r._1._1,r._1._2,r._2._1,r._2._2))

    val preName: Array[String] = Array("eif_id","datadate","predict","realLabel")
    val schema = StructType(preName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfPreLabel = hiveCtx.createDataFrame(preData, schema)
    dfPreLabel.show()
    // insert into hive
    dfPreLabel.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_pre_out as select * from tmp_table")
  }

//  def RDD_to_df()={
//    val dfName: Array[String] = df.columns
//    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>numericalProcessing(r.getAs(x))).toSeq:_*))
//    //DataFrame
//    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
//    val dfFigure = hiveCtx.createDataFrame(data, schema)
//  }
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkTest")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    println("#####################1  得到数据集  训练集+测试集###############")
    val HiveCtx = new HiveContext(sc)
    val train_df=HiveCtx.sql(s"select * from test.alading_train_4_10_good_new where datadate<='2016-10-01' ")
    val test_df=HiveCtx.sql(s"select * from test.alading_train_4_10_good_new where datadate>'2017-01-01' ")
    //    input_df.printSchema()

    println("#####################2  筛选特征：覆盖率 distinct###############")
    val labelField: String = s"label"
    val invalidField: List[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
    "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
    "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
    "datadate_web","num","num_app","Recent_order_product","Recent_order_days","Recent_investment_days","Recent_investment_due_days")


//    get_feature_Select(train_df,test_df,sc,feature_name_df,HiveCtx)

    val singleValueField: Seq[String] =HiveCtx.sql(s"select a.name from test.alading_cover_train a join test.alading_all_value_count" +
    s" b on a.name=b.field_name  where b.distinct_num<=1 ") .map(r => r.getAs[String](0)).collect().toSeq//覆盖率+distinct

    val ignoreField: Seq[String] = invalidField ++ singleValueField
    val featuresFields: Array[String] = train_df.columns.diff(labelField).diff(ignoreField)

    val sqlCtx = new SQLContext(sc)



    //mlib 转换为labelpoint
//    val (targer_train_df,data_train_get)=Get_features_and_todouble(train_df,sc,feature_name_df)
//    val (targer_test_df,data_test_get: RDD[LabeledPoint])=Get_features_and_todouble(test_df,sc,feature_name_df)

//    println("#####################3  特征规则化(标准化 归一化 离散化)+卡方二次选择###############")


    //标准化
//    val train_standard=StandardScaler(targer_train_df)
//    val test_standard=StandardScaler(targer_test_df)
    //归一化
    //val train =Normalizer(data_standard)

    //离散化+得到离散之后的feature_name
//    val dispersed_index: Seq[String] = HiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<100")
//      .map(r => r.getAs[String](0)).collect().toSeq    //对于离散值<targrt 的不进行离散化
//
//    val ChiSqSel_feature_name: Array[String] =get_Disperse_ChiSqSel(targer_train_df,HiveCtx,dispersed_index)
////    //卡方选择
//    val (targer_train_df_new,data_train_get_new)=Get_features_and_todouble(train_df,sc,ChiSqSel_feature_name)
//    val (targer_test_df_new,data_test_get_new)=Get_features_and_todouble(test_df,sc,ChiSqSel_feature_name)
//    println("#####################4  采样###############")
//    val training = mySampling(data_train_get)
//
//
//    //训练模型
//    println("********* 6  模型 ********")
//    val model=train_model(training,data_test_get)
//    val model1=GBDT_model(training,data_test_get)
    }
}

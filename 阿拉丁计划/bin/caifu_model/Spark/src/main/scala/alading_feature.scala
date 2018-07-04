/**
  * Created by wenqiang on 2017/7/21.
  */
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkConf, SparkContext}
import collection._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.{ChiSqSelector, Normalizer, StandardScaler}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.stat.Statistics
import org.json4s.jackson.Json
import scala.collection.mutable.ListBuffer

object AladingFeatureSelect {
  def myToDouble(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
    } catch {
      case ex: NumberFormatException => -999.0
    }
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def dfToDouble(df: DataFrame): RDD[LabeledPoint] = {
    df.map(r => LabeledPoint(myToDouble(r.getAs(0)),Vectors.dense(r.toSeq.drop(1).map(myToDouble).toArray)))
  }

  def dfToDouble(df: DataFrame, hiveCtx:HiveContext): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>myToDouble(r.getAs(x))).toSeq:_*))
    //DataFrame
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = hiveCtx.createDataFrame(data, schema)
    dfFigure
  }

  def myLabledPoint(df: DataFrame): RDD[LabeledPoint] ={
    val featName = df.columns.drop(1)
    val dflabeledPoint: RDD[LabeledPoint] = df.map(r => LabeledPoint(r.getAs[Double](0).toInt,
      Vectors.dense(featName.map(x=>r.getAs[Double](x)))))
    dflabeledPoint
  }

  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  def inforEntropy(target_attribute: Array[Any]): Double = {
    var temp = scala.collection.mutable.HashMap[Any, Int]()
    for (item <- target_attribute) {
      if (temp.contains(item)) {
        temp(item) = temp(item) + 1
      } else {
        temp.put(item, 1)
      }
    }
    var Entropy = 0.0
    for (item <- temp) {
      Entropy += (-1) * (item._2.toDouble / target_attribute.length) * log2(item._2.toDouble / target_attribute.length)
    }
    Entropy
  }

  def inforGain(feature_attribute: RDD[(Any, Any)]): Double = {
    val target: Array[Any] = feature_attribute.map { x => x._2 }.toArray()
    val beforeEntropy: Double = inforEntropy(target)
    //val afterEntropy: Accumulator[Double] = sc.accumulator(0.0)
    var afterEntropy: Double = 0.0
    feature_attribute.groupBy(x => x._1).collect().foreach(x=>afterEntropy += (x._2.size.toDouble / target.length)*inforEntropy(x._2.map(x=>x._2).toArray))
    (beforeEntropy - afterEntropy)
  }

  def inforGainRate(feature_attribute: RDD[(Any, Any)]): Double = {
    val feat: Array[Any] = feature_attribute.map { x => x._1}.toArray()
    val target: Array[Any] = feature_attribute.map { x => x._2 }.toArray()
    val beforeEntropy: Double = inforEntropy(target)
    //val afterEntropy: Accumulator[Double] = sc.accumulator(0.0)
    var afterEntropy: Double = 0.0
    feature_attribute.groupBy(x => x._1).collect().foreach(x=>afterEntropy += (x._2.size.toDouble / feat.length)*inforEntropy(x._2.map(x=>x._2).toArray))
    (beforeEntropy - afterEntropy).toDouble/inforEntropy(feat)
  }

  def correlation(feature_attribute: RDD[(Double, Double)]): Double ={
    val X = feature_attribute.map { x => x._1 }
    val Y = feature_attribute.map { x => x._2 }
    val corr = Statistics.corr(X,Y,"pearson")
    corr
  }

  def myFieldDescribe(df: DataFrame,sc: SparkContext) = {
    // save hdfs
    val dfSize: Int = df.count().toInt
    val coln = df.columns
    val dfDescrIn: SchemaRDD = df.describe(coln.toSeq:_*)
    // save local ([[summary,count...]...])
    val columnName: Array[String] = dfDescrIn.columns
    val dataTranspose: Vector[Seq[String]] = columnName.map(c =>
      (Array(c) ++ (dfDescrIn.select(c).map(r => r.getAs[String](0)).collect())).toSeq).toVector
    //val outFile = "/user/dev/ml/alading_cover_wq"
    val outFile: String = "file:///C://Users/Administrator.20161229-121505/Desktop/tmp_result/alading_cover_train_wq"
    sc.parallelize(dataTranspose.toVector)
      .repartition(1).saveAsTextFile(outFile)
    print ("Save completed")
  }

  def myFieldDescribe(df: DataFrame,sc: SparkContext,hiveCtx: HiveContext) = {
    // save hive
    val dfSize: Double = df.count().toDouble
    assert(dfSize>0)
    val columnName: Array[String] = df.columns
    val dfDescrIn: SchemaRDD = df.describe(columnName.toSeq:_*)
    dfDescrIn.show()

    // save DataFrame test
    val fieldName: Array[String] = dfDescrIn.columns.drop(1)
    val dataTranspose: Array[Array[String]] = fieldName.map(c => (Array(c)++(dfDescrIn.select(c).map(r => r.getAs[String](0)).collect())))

    // Trans to DataFrame
    val dfName: Array[String] = Array("field_name")++(dfDescrIn.map(x =>x.getAs[String](0)).collect())
    val data: RDD[Row] = sc.parallelize(dataTranspose.map(r => Row(r.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescrOut = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)
    dfDescrOut.show()

    // insert into hive
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_describe_info_wq as select * from tmp_table")
  }

  def myValueCount(df: DataFrame,sc: SparkContext,hiveCtx: HiveContext) = {
    df.registerTempTable("testTable")
    val sqlText: String = s"select count(distinct(&&field&&)) from testTable"
    val dataSql: Array[List[String]] = df.columns.map(x =>
      List(x,hiveCtx.sql(sqlText.replace("&&field&&",x)).first().getAs[Long](0).toString))

    val dfName: Array[String] = Array("field_name","distinct_count")
    val data: RDD[Row] = sc.parallelize(dataSql.map(x =>Row(x.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfValueCount = hiveCtx.createDataFrame(data, schema)
    dfValueCount.show()

    // insert into hive

    dfValueCount.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_value_count_wq as select * from tmp_table")
  }

  def myInforGainRate(df1:DataFrame,sc: SparkContext,hiveCtx: HiveContext) = {
    val ignored: Seq[String] = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<=1")
      .map(r => r.getAs[String](0)).collect().toSeq
    val target = df1.columns(0)
    val colName: Array[String] = df1.columns.drop(1).diff(ignored)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    colName.foreach(println)
    val df = df1.select(target,colName.toSeq:_*)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    df.printSchema()
    val labelName: String = df.columns(0)
    val featName: Array[String] = df.columns.drop(1)

    // test
    val inforGainRateValue: ListBuffer[Double] = new ListBuffer[Double]
    for (field <- featName) {
      val data: RDD[(Any, Any)] = df.map(r => (r.getAs[Any](field), r.getAs[Any](labelName)))
      data.count()
      inforGainRateValue.append(inforGainRate(data))
    }
    print ("**********************3*******************")
    // test end
    // test
    /*
    val data1: RDD[Array[(Any, Any)]] = df.map (r =>
      featName.map(field=>(r.getAs[Any](field),r.getAs[Any](labelName))))
    data1.foreach(x=> inforGainRate(x))
    val aa: RDD[(Any, Any)] =data1.map(x=>x(0))
    */
    // test end

    val inforData: Array[List[String]] = featName.zip(inforGainRateValue.map(x => x.toString)).map(x=>List(x._1,x._2))
    val dfName: Array[String] = Array("field_name","inforGainRate")
    val data: RDD[Row] = sc.parallelize(inforData.map(x =>Row(x.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfInforGain = hiveCtx.createDataFrame(data, schema)
    dfInforGain.show()
    println("********************************** to dataframe *************************************")
    // insert into hive
    dfInforGain.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_inforGrainRate_wq as select * from tmp_table")
  }

  def myCorr(df1:DataFrame,sc: SparkContext,hiveCtx: HiveContext) = {
    val ignored: Seq[String] = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<=1")
      .map(r => r.getAs[String](0)).collect().toSeq
    val target = df1.columns(0)
    val colName: Array[String] = df1.columns.drop(1).diff(ignored)
    val df2 = df1.select(target,colName.toSeq:_*)
    val df = dfToDouble(df2, hiveCtx)

    val labelName: String = df.columns(0)
    val featName: Array[String] = df.columns.drop(1)
    //val combinations: Array[RDD[(Double, Double)]] = featName.map(x =>
    //df.map(r => (r.getAs[Double](x),r.getAs[Double](labelName))))
    //val inforGainRateValue: Array[Double] = combinations.map(x => correlation(x))
    val inforGainRateValue: ListBuffer[Double] = new ListBuffer[Double]
    for (field <- featName) {
      val data: RDD[(Double, Double)] = df.map(r => (r.getAs[Double](field),r.getAs[Double](labelName)))
      inforGainRateValue.append(correlation(data))
    }
    val inforData: Array[List[String]] = featName.zip(inforGainRateValue.map(x => x.toString)).map(x=>List(x._1,x._2))

    val dfName: Array[String] = Array("field_name","correlation")
    val data: RDD[Row] = sc.parallelize(inforData.map(x =>Row(x.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfValueCount = hiveCtx.createDataFrame(data, schema)
    dfValueCount.show()

    // insert into hive
    dfValueCount.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_correlation_wq as select * from tmp_table")
  }

  def myChiSqSelector(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val selector = new ChiSqSelector(300)
    val transformer = selector.fit(data)
    println("The selectedFeatures's index，counting from 0")
    println(transformer.selectedFeatures.mkString("~"))
    data.map(lp => LabeledPoint(lp.label, transformer.transform(lp.features)))
  }

  def myChiSqSelector(df1: DataFrame,sc: SparkContext,hiveCtx: HiveContext): SchemaRDD = {
    val ignored: Seq[String] =  hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<=1")
      .map(r => r.getAs[String](0)).collect().toSeq
    val target: String = df1.columns(0)
    val colName: Array[String] = df1.columns.drop(1).diff(ignored)
    val df2 = df1.select(target,colName.toSeq:_*)

    val df = dfToDouble(df2, hiveCtx)
    print("**************** df ******************")
    df.printSchema()



    val dfName: Array[String] = df.columns
    val label: String = dfName(0)
    val featIn: Array[String] = dfName.drop(1)
    // Disc + Series
    val ignoreDisc: Seq[String] = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<500")
      .map(r => r.getAs[String](0)).collect().toSeq

    val featSeries: Array[String] =  featIn.diff(ignoreDisc)
    val featDisc = featIn.diff(featSeries.toSeq)
    val featOut: Array[String] = featDisc++featSeries
    // test start
    println("************************** 1 **********************************")
    println(featOut.length)
    //test end

    // Discretizer
    import org.apache.spark.ml.feature.QuantileDiscretizer
    import org.apache.spark.ml.Pipeline
    val discretizers = featSeries.map(c =>
      new QuantileDiscretizer().setInputCol(c).setOutputCol(s"${c}_disc").setNumBuckets(100))
    val pipeline: Pipeline = new Pipeline().setStages(discretizers)
    val dfDisc = pipeline.fit(df).transform(df)
    print("*********** 离散化 ***********")
    dfDisc.printSchema()
    dfDisc.show(3)

    // ChiSqSelector
    val featChi: Array[String] = dfDisc.columns.diff((label++featSeries).toSeq)
    val dataChi: RDD[LabeledPoint] = dfDisc.map(r =>
      LabeledPoint(r.getAs[Double](label).toInt, Vectors.dense(featChi.map(x => r.getAs[Double](x)))))
    val selector = new ChiSqSelector(150)
    val transformer = selector.fit(dataChi)
    val selectName: Array[String] = transformer.selectedFeatures.map(x => featOut(x))

    print("********* 卡方选择 **********")
    println("The selectedFeatures's Name")
    println(selectName.mkString(" ~ "))

    val dfName2: Array[String] = Array("field_ChiSqSeler")
    val data: RDD[Row] = sc.parallelize(selectName.map(x =>Row(x)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName2.map(fieldName => StructField(fieldName, StringType, true)))
    val dfValueCount = hiveCtx.createDataFrame(data, schema)
    dfValueCount.show()

    // insert into hive
    dfValueCount.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_ChiSqSelect_wq_new as select * from tmp_table")
  }

  def loadData(tableName: String, hiveCtx: HiveContext): SchemaRDD = {
    val train: SchemaRDD = hiveCtx.sql(s"select * from test.${tableName} where datadate<='2016-10-01'")
    //val test = hiveCtx.sql(s"select * from test.${tableName} where datadate>='2017-01-01'")
    //List(train,test)
    train
  }

  def loadJsonData(fileName: String, hiveCtx: HiveContext) = {
    val jsons = hiveCtx.read.json(fileName)
    jsons.registerTempTable("testTable")
    hiveCtx.sql(s"select * from testTable")
  }

  def preProcess(df: DataFrame,hiveCtx:HiveContext): SchemaRDD = {
    val ignored: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app")
    val targetName: String = ("label")
    val featName: Array[String] = df.columns.diff(ignored)
    val dfData: SchemaRDD = df.select(targetName,featName.toSeq:_*)
    dfData
  }



  // test -start 0
  def myChiSqSelector(df: DataFrame): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val label: String = dfName(0)
    val featIn: Array[String] = dfName.drop(1)
    // Disc + Series
    val featSeries = Array(
      "all_investment",
      "year_investment",
      "half_year_investment",
      "three_months_investment",
      "all_investment_avg",
      "year_investment_avg",
      "half_year_investment_avg",
      "three_months_investment_avg",
      "product_terms_avg",
      "year_product_terms_avg",
      "half_year_product_terms_avg",
      "three_months_product_terms_avg",
      "recent_investment_days",
      "recent_investment_amount",
      "recent_investment_due_days",
      "recent_investment_due_amount",
      "maximum_amount",
      "minimum_amount",
      "avg_communicate_len_hour_14_day",
      "avg_communicate_len_hour_1_month",
      "avg_communicate_len_hour_2_month",
      "communicate_len_hour_3_month",
      "avg_communicate_len_hour_3_month",
      "u_id_app",
      "app_pv_all",
      "app_pv_7_days",
      "app_pv_14_days",
      "app_pv_1_months",
      "app_pv_2_months",
      "app_pv_3_months",
      "app_pv_6_months",
      "u_id_web",
      "web_pv_all",
      "pv_7_days",
      "pv_14_days",
      "pv_1_month",
      "pv_2_month",
      "pv_3_month",
      "pv_6_month",
      "all_investment_cnt",
      "avg_communicate_cnt_1_month",
      "avg_communicate_cnt_2_month",
      "avg_communicate_cnt_3_month"
    )
    val featDisc = featIn.diff(featSeries.toSeq)
    val featOut = featDisc++featSeries

    // Discretizer
    import org.apache.spark.ml.feature.QuantileDiscretizer
    import org.apache.spark.ml.Pipeline
    val discretizers = featSeries.map(c =>
      new QuantileDiscretizer().setInputCol(c).setOutputCol(s"${c}_disc").setNumBuckets(100))
    val pipeline: Pipeline = new Pipeline().setStages(discretizers)
    val dfDisc = pipeline.fit(df).transform(df)
    print("*********** 离散化 ***********")
    dfDisc.printSchema()
    dfDisc.show(3)

    // ChiSqSelector
    val featChi: Array[String] = dfDisc.columns.diff((label++featSeries).toSeq)
    val dataChi: RDD[LabeledPoint] = dfDisc.map(r =>
      LabeledPoint(r.getAs[Double](label).toInt, Vectors.dense(featChi.map(x => r.getAs[Double](x)))))
    val selector = new ChiSqSelector(200)
    val transformer = selector.fit(dataChi)
    val selectName: Array[String] = transformer.selectedFeatures.map(x => featOut(x))
    print("********* 卡方选择 **********")
    println("The selectedFeatures's Name")
    println(selectName.mkString(" ~ "))
    df.select(label, selectName.toSeq: _*)
  }

  def myCountSelector(df: DataFrame): SchemaRDD = {
    val columnName: Array[String] = df.columns
    val all_length = df.count()
    // #describe
    //val cover: Array[Double] = columnName.map(x=>df.describe(x).first().get(1).toString.toInt).map(x=>x*1.0/all_length)
    // #agg
    val cover: Array[Double] = columnName.map(r =>
      df.agg(count(r)).first().get(0).toString.toInt).map(x => x * 1.0 / all_length)
    val coverdescribe: Array[(String, Double)] = columnName.zip(cover)

    //coverdescribe.foreach(println)
    val coverselect: Array[(String, Double)] = coverdescribe.filter(x => x._2 > 0.1)
    df.select(coverdescribe.map(x => x._1).map(col).toSeq: _*)
  }

  def staticSelect(hiveCtx: HiveContext ): Array[String] ={
    val ignore1: Array[String] = hiveCtx.sql(s"select name from test.alading_cover_train where cover_rate<=0.001")
      .map(r => r.getAs[String](0)).collect()
    val ignore2: Array[String] = hiveCtx.sql(s"select field_name from test.alading_all_value_count where distinct_num<=1")
      .map(r => r.getAs[String](0)).collect()
    val ignore = ignore1 ++ ignore2.diff(ignore1.toSeq)

    ignore
  }

  def myInforGainRate2(df:DataFrame,sc: SparkContext,hiveCtx: HiveContext) = {
    val labelName: String = df.columns(0)
    val featName: Array[String] = df.columns.drop(1)
    val combinations: Array[RDD[(Any, Any)]] = featName.map(x => df.map(r => (r.getAs[Any](x),r.getAs[Any](labelName))))
    val inforGainRateValue: Array[(Double, Double)] = combinations.map{ x =>
      val y = x.map(r =>(r._1.toString.toDouble,r._2.toString.toDouble))
      (inforGainRate(x),correlation(y))
    }
    val inforData: Array[List[String]] = featName.zip(inforGainRateValue)
      .map(x=>List(x._1,x._2._1.toString,x._2._1.toString))

    val dfName: Array[String] = Array("field_name","inforGainRate","correlation")
    val data: RDD[Row] = sc.parallelize(inforData.map(x =>Row(x.toSeq:_*)).toVector)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfValueCount = hiveCtx.createDataFrame(data, schema)
    dfValueCount.show()

    // insert into hive
    /*
    dfValueCount.registerTempTable("tmp_table")
    hiveCtx.sql("use test")
    hiveCtx.sql("create table alading_inforGrainRate_wq as select * from tmp_table")
    */
  }

  // test -end 0

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("aladingFeature")//.setMaster("local")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("WARN")

    // Load Data
    //val file_name = "./data/aaa.json"
    //val trInput = loadJsonData(file_name,hiveCtx)
    val trInput = loadData("alading_train_4_10_new", hiveCtx)

    // Preprocessor
    val dfData = preProcess(trInput,hiveCtx)
    //val dfData = trInput
    println("***** Table structure *****")
    dfData.printSchema()
    dfData.show(2)

    // 1.Static - describe
    //myFieldDescribe(dfData,sc,hiveCtx)
    // 2.Static - count(distinct)
    //myValueCount(dfData,sc,hiveCtx)
    // 3.Static - informationGain & correlation
    myInforGainRate(dfData,sc,hiveCtx)
    // 4.Static - correlation
    //myCorr(dfData,sc,hiveCtx)
    // 5.Static - L2

    // 6.Static - ChiSqSelect
    //myChiSqSelector(dfData,sc,hiveCtx)
  }
}
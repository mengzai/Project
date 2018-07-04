import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.Bucketizer
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
import org.apache.spark.mllib.util.{MLUtils, Saveable}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.util.random.RandomSampler
import org.apache.spark.mllib.feature.{ChiSqSelector, ChiSqSelectorModel, Normalizer, StandardScaler}

import collection._
import org.apache.spark.mllib.stat.Statistics
import org.json4s.jackson.Json
/*
 def inforGainRate(sc: SparkContext, feature_attribute: RDD[(Double, Double)]): (Double, Double) = {
   val target = feature_attribute.map { x => x._2 }.toArray()
   val Entropy1 = myInforEntropy(target)

   val all_Entropy = sc.accumulator(0.0)
   feature_attribute.groupBy(x => x._1).foreach { x => all_Entropy += (x._2.size.toDouble / target.length) * myInforEntropy(x._2.map(x => x._2).toArray)
   }

   val X = feature_attribute.map { x => x._1 }
   val Y = feature_attribute.map { x => x._2 }

   val correlation: Double = Statistics.corr(X, Y, "pearson")
   /*    // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
   // If a method is not specified, Pearson's method will be used by default.
   val correlMatrix: Matrix = Statistics.corr(data, "pearson")*/

   //    println(Entropy1)
   //    println(all_Entropy.value)
   ((Entropy1 - all_Entropy.value).toDouble/myInforEntropy(X.toArray()), correlation)
 }
 */
object AladingModel4 {
  def myToDouble(x: Any): Double = x match {
    case s: String => try {
      s.toDouble
    } catch {
      case ex: NumberFormatException => -999.0
    }
    case jn: java.lang.Number => jn.doubleValue()
    case _ => -999.0
  }

  def dfToDouble(df: DataFrame, sc: SparkContext) = {
    val dfName: Array[String] = df.columns
    val data: RDD[Row] = df.map(r =>Row(dfName.map(x =>myToDouble(r.getAs(x))).toSeq:_*))
    //trans to dataframe
    val sqlContext = new SQLContext(sc)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, DoubleType, true)))
    val dfFigure = sqlContext.createDataFrame(data, schema)
    dfFigure
  }

  def myStandardScaler(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val scalar = new StandardScaler(withMean = true, withStd = true).fit(data.map(r => r.features))
    data.map(lp => LabeledPoint(lp.label, scalar.transform(lp.features)))
  }

  def myChiSqSelector(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val selector = new ChiSqSelector(300)
    val transformer = selector.fit(data)
    println("The selectedFeatures's index，counting from 0")
    println(transformer.selectedFeatures.mkString("~"))
    data.map(lp => LabeledPoint(lp.label, transformer.transform(lp.features)))
  }

  def myChiSqSelector(df: DataFrame): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val label: String = dfName(0)
    val featIn: Array[String] = dfName.drop(0)
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
    val featDisc = featIn.diff(featSeries)
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
    val featChi: Array[String] = dfDisc.columns.diff(label++featSeries)
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

/*
  //test -start
  def myChiSqSelector(df: DataFrame): SchemaRDD = {
    val dfName: Array[String] = df.columns
    val label: String = dfName(0)
    val feat: Array[String] = dfName.drop(0)
    // Discretizer
    import org.apache.spark.ml.feature.QuantileDiscretizer
    // start

    val disctetizers = dfName.map(c =>new QuantileDiscretizer().setInputCol(c).setOutputCol(s"${c}_disc").setNumBuckets(2).fit(df))
    val pipeline = new Pipeline().setStages(discretizers)
    val dfDisc = mydisctetizers.fit(df).transform(df)
    print("*********** 离散化 ***********")
    dfDisc.printSchema()
    dfDisc.show(3)
    // end

    // ChiSqSelector
    val dataChi: RDD[LabeledPoint] = dfDisc.map(r =>
      LabeledPoint(r.getAs[Double](label).toInt, Vectors.dense(feat.map(x => r.getAs[Double](x+"_disc")))))
    val selector = new ChiSqSelector(200)
    val transformer = selector.fit(dataChi)
    val selectName: Array[String] = transformer.selectedFeatures.map(x => feat(x))
    print("********* 卡方选择 **********")
    println("The selectedFeatures's Name")
    println(selectName.mkString(" ~ "))
    df.select(label, selectName.toSeq: _*)
  }
  //test-end
  */
  def myLabledPoint(df: DataFrame): RDD[LabeledPoint] ={
    val featName = df.columns.drop(0)
    val dflabeledPoint: RDD[LabeledPoint] = df.map(r => LabeledPoint(r.getAs[Double](0).toInt,
      Vectors.dense(featName.map(x=>r.getAs[Double](x)))))
    dflabeledPoint
  }

  def myNormalizer(data: RDD[LabeledPoint]) = {
    val l2 = new Normalizer(2)
    l2.transform(data.map(x => x.features))
    data.map(x => LabeledPoint(x.label, x.features))
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

  def log2(x: Double): Double = scala.math.log(x) / scala.math.log(2)

  def inforEntropy(target_attribute: Array[Double]): Double = {
    var temp = scala.collection.mutable.HashMap[Double, Int]()
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

  def myInforEntropy(df:DataFrame): SchemaRDD = {
    val columnName: Array[String] = df.columns
    val infoValue: Array[Double] = columnName.map(col =>
      inforEntropy(df.select(col).map(x => myToDouble(x.getAs(0))).collect()))
    val infodescribe: Array[(String, Double)] = columnName.zip(infoValue)
    //coverdescribe.foreach(println)
    val infoselect: Array[(String, Double)] = infodescribe.filter(x=> x._2>0.3)
    df.select(infoselect.map(x => x._1).map(col).toSeq:_*)
  }

  def loadData(tableName: String, hiveCtx: HiveContext) = {
    hiveCtx.sql(s"select * from test.${tableName} where datadate<= '2016-10-01'")
  }

  def loadJsonData(fileName: String, hiveCtx: HiveContext) = {
    val jsons = hiveCtx.read.json(fileName)
    jsons.registerTempTable("testTable")
    hiveCtx.sql(s"select * from testTable")
  }

  def preProcess(df: DataFrame,sc:SparkContext): RDD[LabeledPoint] = {
    //val targetName: String = ("name")
    //val ignored: Seq[String] = List("name")
    val targetName: String = ("label")
    val ignored: Seq[String] = List("label","ecif_id","datadate","ecif_id_trade","datadate_trade","ecif_id_activity",
      "ecif_id_communicate","datadate_communicate","ecif_id_kyc","datadate_kyc",
      "ecif_id_order","datadate_order","u_id_app","datadate_app","u_id_web",
      "datadate_web","num","num_app")
    val featName: Array[String] = df.columns.diff(ignored)
    val selectName: Array[String] = Array("all_investment",
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
      "recent_investment_product",
      "recent_investment_due_days",
      "recent_investment_due_amount",
      "recent_investment_due_product",
      "maximum_amount",
      "minimum_amount",
      "activity_avg_3_month",
      "fact_activity_avg_3_month",
      "activity_duration_avg",
      "activity_duration_avg_year",
      "activity_duration_avg_half_year",
      "activity_duration_avg_3_month",
      "activity_duration_avg_2_month",
      "activity_duration_avg_1_month",
      "new_customer_transform",
      "old_customer_maintain",
      "brand_promote",
      "road_show",
      "business_show",
      "exhibite",
      "acknowledge_activity",
      "financial_lectures",
      "brand_activity",
      "get_new_customer_year",
      "new_customer_transform_year",
      "old_customer_maintain_year",
      "brand_promote_year",
      "road_show_year",
      "business_show_year",
      "exhibite_year",
      "acknowledge_activity_year",
      "financial_lectures_year",
      "brand_activity_year",
      "get_new_customer_half_year",
      "new_customer_transform_half_year",
      "old_customer_maintain_half_year",
      "brand_promote_half_year",
      "road_show_half_year",
      "business_show_half_year",
      "exhibite_half_year",
      "acknowledge_activity_half_year",
      "financial_lectures_half_year",
      "brand_activity_half_year",
      "get_new_customer_3_month",
      "new_customer_transform_3_month",
      "old_customer_maintain_3_month",
      "brand_promote_3_month",
      "road_show_3_month",
      "business_show_3_month",
      "exhibite_3_month",
      "acknowledge_activity_3_month",
      "financial_lectures_3_month",
      "brand_activity_3_month",
      "get_new_customer_2_month",
      "new_customer_transform_2_month",
      "old_customer_maintain_2_month",
      "brand_promote_2_month",
      "road_show_2_month",
      "business_show_2_month",
      "exhibite_2_month",
      "acknowledge_activity_2_month",
      "financial_lectures_2_month",
      "brand_activity_2_month",
      "get_new_customer_1_month",
      "new_customer_transform_1_month",
      "old_customer_maintain_1_month",
      "brand_promote_1_month",
      "road_show_1_month",
      "business_show_1_month",
      "exhibite_1_month",
      "acknowledge_activity_1_month",
      "financial_lectures_1_month",
      "brand_activity_1_month",
      "communicate_len_hour_1_day",
      "avg_communicate_len_hour_1_day",
      "communicate_len_hour_7_day",
      "avg_communicate_len_hour_7_day",
      "communicate_len_hour_14_day",
      "avg_communicate_len_hour_14_day",
      "communicate_len_hour_1_month",
      "avg_communicate_len_hour_1_month",
      "communicate_len_hour_2_month",
      "avg_communicate_len_hour_2_month",
      "communicate_len_hour_3_month",
      "avg_communicate_len_hour_3_month",
      "communicate_len_hour_1_day_mode_1",
      "communicate_len_hour_1_day_mode_2",
      "communicate_len_hour_1_day_mode_3",
      "communicate_len_hour_1_day_mode_4",
      "communicate_len_hour_1_day_mode_5",
      "communicate_len_hour_1_day_mode_6",
      "communicate_len_hour_7_day_mode_1",
      "communicate_len_hour_7_day_mode_2",
      "communicate_len_hour_7_day_mode_3",
      "communicate_len_hour_7_day_mode_4",
      "communicate_len_hour_7_day_mode_5",
      "communicate_len_hour_7_day_mode_6",
      "communicate_len_hour_14_day_mode_1",
      "communicate_len_hour_14_day_mode_2",
      "communicate_len_hour_14_day_mode_3",
      "communicate_len_hour_14_day_mode_4",
      "communicate_len_hour_14_day_mode_5",
      "communicate_len_hour_14_day_mode_6",
      "communicate_len_hour_1_month_mode_1",
      "communicate_len_hour_1_month_mode_2",
      "communicate_len_hour_1_month_mode_3",
      "communicate_len_hour_1_month_mode_4",
      "communicate_len_hour_1_month_mode_5",
      "communicate_len_hour_1_month_mode_6",
      "communicate_len_hour_2_month_mode_1",
      "communicate_len_hour_2_month_mode_2",
      "communicate_len_hour_2_month_mode_3",
      "communicate_len_hour_2_month_mode_4",
      "communicate_len_hour_2_month_mode_5",
      "communicate_len_hour_2_month_mode_6",
      "communicate_len_hour_3_month_mode_1",
      "communicate_len_hour_3_month_mode_2",
      "communicate_len_hour_3_month_mode_3",
      "communicate_len_hour_3_month_mode_4",
      "communicate_len_hour_3_month_mode_5",
      "communicate_len_hour_3_month_mode_6",
      "last_kyc_score",
      "last_investabel",
      "last_communicate_score",
      "last_roi",
      "last_accept_loss",
      "last_investment_term",
      "invest_bank_financing",
      "invest_money_fund",
      "invest_public_fund",
      "invest_house_property",
      "invest_trust",
      "invest_p3p",
      "invest_shares",
      "invest_private_fund",
      "invest_vc",
      "last_willingness1",
      "last_willingness2",
      "last_willingness3",
      "last_willingness4",
      "last_willingness5",
      "last_willingness6",
      "three_months_investabel_avg",
      "three_months_investabel_max",
      "three_months_investabel_median",
      "three_months_investabel_min",
      "half_year_investabel_avg",
      "half_year_investabel_max",
      "half_year_investabel_median",
      "half_year_investabel_min",
      "year_investabel_avg",
      "year_investabel_max",
      "year_investabel_median",
      "year_investabel_min",
      "all_investabel_avg",
      "all_investabel_max",
      "all_investabel_median",
      "all_investabel_min",
      "order_recent_order_product",
      "order_recent_order_days",
      "u_id_app",
      "app_pv_all",
      "app_pv_1_days",
      "app_pv_7_days",
      "app_pv_14_days",
      "app_pv_1_months",
      "app_pv_2_months",
      "app_pv_3_months",
      "app_pv_6_months",
      "u_id_web",
      "web_pv_all",
      "pv_1_days",
      "pv_7_days",
      "pv_14_days",
      "pv_1_month",
      "pv_2_month",
      "pv_3_month",
      "pv_6_month",
      "all_investment_cnt",
      "year_investment_cnt",
      "half_year_investment_cnt",
      "three_months_investment_cnt",
      "product_category_cnt",
      "year_product_category_cnt",
      "half_year_product_category_cnt",
      "three_months_product_category_cnt",
      "private_equity_cnt",
      "capital_market_cnt",
      "p2p_cnt",
      "foundation_cnt",
      "insurance_cnt",
      "year_private_equity_cnt",
      "year_capital_market_cnt",
      "year_p2p_cnt",
      "year_foundation_cnt",
      "year_insurance_cnt",
      "half_year_private_equity_cnt",
      "half_year_capital_market_cnt",
      "half_year_p2p_cnt",
      "half_year_foundation_cnt",
      "half_year_insurance_cnt",
      "three_months_private_equity_cnt",
      "three_months_capital_market_cnt",
      "three_months_p2p_cnt",
      "three_months_foundation_cnt",
      "three_months_insurance_cnt",
      "activity_cnt",
      "fact_activity_cnt",
      "activity_cnt_year",
      "fact_activity_cnt_year",
      "activity_cnt_half_year",
      "fact_activity_cnt_half_year",
      "activity_cnt_3_month",
      "fact_activity_cnt_3_month",
      "activity_cnt_2_month",
      "fact_activity_cnt_2_month",
      "activity_cnt_1_month",
      "fact_activity_cnt_1_month",
      "activity_duration_cnt",
      "activity_duration_cnt_year",
      "activity_duration_cnt_half_year",
      "activity_duration_cnt_3_month",
      "activity_duration_cnt_2_month",
      "activity_duration_cnt_1_month",
      "communicate_cnt_1_day",
      "avg_communicate_cnt_1_day",
      "communicate_cnt_7_day",
      "avg_communicate_cnt_7_day",
      "communicate_cnt_14_day",
      "avg_communicate_cnt_14_day",
      "communicate_cnt_1_month",
      "avg_communicate_cnt_1_month",
      "communicate_cnt_2_month",
      "avg_communicate_cnt_2_month",
      "communicate_cnt_3_month",
      "avg_communicate_cnt_3_month",
      "communicate_cnt_1_day_mode_4",
      "communicate_cnt_7_day_mode_4",
      "communicate_cnt_14_day_mode_4",
      "communicate_cnt_1_month_mode_4",
      "communicate_cnt_2_month_mode_4",
      "communicate_cnt_3_month_mode_4",
      "three_months_kyc_invalid_cnt",
      "three_months_kyc_valid_cnt",
      "three_months_invitation_null_cnt",
      "three_months_invitation_failed_cnt",
      "three_months_invitation_successful_cnt",
      "three_months_invitation_other_cnt",
      "three_months_kyc_cnt",
      "three_months_investment_term1_cnt",
      "three_months_investment_term2_cnt",
      "three_months_investment_term3_cnt",
      "three_months_investment_term4_cnt",
      "half_year_kyc_invalid_cnt",
      "half_year_kyc_valid_cnt",
      "half_year_invitation_null_cnt",
      "half_year_invitation_failed_cnt",
      "half_year_invitation_successful_cnt",
      "half_year_invitation_other_cnt",
      "half_year_kyc_cnt",
      "half_year_investment_term1_cnt",
      "half_year_investment_term2_cnt",
      "half_year_investment_term3_cnt",
      "half_year_investment_term4_cnt",
      "year_kyc_invalid_cnt",
      "year_kyc_valid_cnt",
      "year_invitation_null_cnt",
      "year_invitation_failed_cnt",
      "year_invitation_successful_cnt",
      "year_invitation_other_cnt",
      "year_kyc_cnt",
      "year_investment_term1_cnt",
      "year_investment_term2_cnt",
      "year_investment_term3_cnt",
      "year_investment_term4_cnt",
      "all_kyc_invalid_cnt",
      "all_kyc_valid_cnt",
      "all_invitation_null_cnt",
      "all_invitation_failed_cnt",
      "all_invitation_successful_cnt",
      "all_invitation_other_cnt",
      "all_kyc_cnt",
      "all_investment_term1_cnt",
      "all_investment_term2_cnt",
      "all_investment_term3_cnt",
      "all_investment_term4_cnt",
      "all_willingness1_cnt",
      "all_willingness2_cnt",
      "all_willingness3_cnt",
      "all_willingness4_cnt",
      "all_willingness5_cnt",
      "all_willingness6_cnt",
      "all_order_cnt",
      "order_product_category_cnt",
      "order_year_cnt",
      "order_year_product_category_cnt",
      "order_half_year_cnt",
      "order_half_year_product_category_cnt",
      "order_three_months_cnt",
      "order_three_months_product_category_cnt",
      "order_private_equity_cnt",
      "order_capital_market_cnt",
      "order_foundation_cnt",
      "order_fixed_income_cnt",
      "order_year_private_equity_cnt",
      "order_year_capital_market_cnt",
      "order_year_foundation_cnt",
      "order_year_fixed_income_cnt",
      "order_half_year_private_equity_cnt",
      "order_half_year_capital_market_cnt",
      "order_half_year_foundation_cnt",
      "order_half_year_fixed_income_cnt",
      "order_three_months_private_equity_cnt",
      "order_three_months_capital_market_cnt",
      "order_three_months_foundation_cnt",
      "three_months_fixed_income_cnt"
    )
    val dfData = df.select(targetName,selectName.toSeq:_*)

    // 1.trans to figure
    val dfFigure: SchemaRDD = dfToDouble(dfData,sc)
    print("*********** 2.1 转换为数值类型 ***********")
    dfFigure.show(3)

    // 2.count desctibe
    //val dfCountSelect: SchemaRDD = myCountSelector(dfData)
    val dfCountSelect: SchemaRDD = dfFigure

    // 3.ChiSqSelector
    //val dfChi: SchemaRDD = myChiSqSelector(dfCountSelect)
    val dfChi = dfCountSelect
    print("************** 2.2 卡方选则 ************")
    dfChi.show(3)

    // 4.trans to LabeledPoint
    val dfLabeledPoint: RDD[LabeledPoint] = myLabledPoint(dfChi)

    // 3.StanderdScaler
    val dfScaler: RDD[LabeledPoint] = myStandardScaler(dfLabeledPoint)

    // del - 4.ChiSqSelector
    //val dfChiSq = myChiSqSelector(df_Scaler)

    dfScaler
  }

  def mySampling(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val dataSample: RDD[(Int, LabeledPoint)] = data.map(row => {
      if (row.label == 1.0)
        (row, 1)
      else (row, 2)
    }).map(x => (x._2, x._1))
    val fractions: Map[Int, Double] = (List((1, 1.0), (2, 0.05))).toMap
    val approSample = dataSample.sampleByKey(withReplacement = true, fractions, 0)
    //approxSample.foreach(println)
    approSample.map(x => x._2)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("aladingModel")//.setMaster("local");
    val sc = new SparkContext(conf)
    sc.setLogLevel("INFO")

    // 1.Load Data
    val hiveCtx = new HiveContext(sc)
    //val file_name = "./data/aaa.json"
    //val trInput = loadJsonData(file_name, hiveCtx)
    val trInput = loadData("alading_train_4_10_new", hiveCtx)
    println("***** 1.表结构 *****")
    trInput.printSchema()

    // 2.preprocess
    val train: RDD[LabeledPoint] = preProcess(trInput,sc)
    println("***** 2.预处理数据 *****")
    train.take(20).foreach(println)
    println(train.count())

    // 3.Sampling
    val sampling_data = mySampling(train)
    println("********* 3.采样 ********")
    sampling_data.foreach(println)
    val splits = sampling_data.randomSplit(Array(0.8, 0.2))
    val data_train = splits(0)
    val data_test = splits(1)

    // 4.模型训练
    println("******* 4.开始训练 *******")
    // 1.NaiveBayes
    import org.apache.spark.mllib.classification.NaiveBayes
    //val model = NaiveBayes.train(input = data_train, lambda = 1.0, modelType = "multinomial")

    // 2.LR - LogisticRegressionWithSGD
    import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
    //val model = LogisticRegressionWithSGD.train(input = data_train, numIterations = 2)

    // 3.LR - LogisticRegressionWithLBFGS
    import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
    //val model = new LogisticRegressionWithLBFGS().setNumClasses(numClasses = 10).run(input = data_train)

    // 4.SVMWithSGD
    import org.apache.spark.mllib.classification.SVMWithSGD
    //val model = SVMWithSGD.train(input = data_train, numIterations = 2)

    // 5.DecisionTree
    import org.apache.spark.mllib.tree.DecisionTree
    //val model:DecisionTreeModel = DecisionTree.trainClassifier(input = data_train,numClasses = 2,
    //categoricalFeaturesInfo = collection.immutable.Map[Int,Int](),impurity = "gini", maxDepth = 5, maxBins = 32)

    // 6.RandomForest
    import org.apache.spark.mllib.tree.RandomForest
    val model = RandomForest.trainClassifier(input=data_train, numClasses = 2,
      categoricalFeaturesInfo = collection.immutable.Map[Int,Int](),numTrees = 10,
      featureSubsetStrategy = "auto", impurity = "gini", maxDepth =5, maxBins =32,seed = 123)

    // 7.GradientBoostedTrees
    import org.apache.spark.mllib.tree.GradientBoostedTrees
    import org.apache.spark.mllib.tree.configuration.BoostingStrategy
    //val boostingStrategy = BoostingStrategy.defaultParams("Classification")//default param
    //val model = GradientBoostedTrees.train(input = data_train, boostingStrategy = boostingStrategy)

    // 5. 预测
    println("***** 5.开始预测 *****")
    val predictionAndLabels = data_test.map{ case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    println ("***** 预测输出/真实标签（前20) *****")
    predictionAndLabels.take(20).foreach(println)

    // 6.ROC评估
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val roc = metrics.roc
    val auROC = metrics.areaUnderROC
    println ("***** 6.评估 - AUC *****")
    println("Area under ROC = " + auROC)
  }
}

package cn.creditease.invest.test
/**
  * Created by Administrator on 2017/8/17.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

object describe1 {
  def fieldProcessing(hiveCtx:HiveContext,df:DataFrame) = {
    val name: Array[String] = df.columns
    val data: RDD[Row] = df.map(r=>r.toSeq.map(x=>
      if(x == null) null else if (x.toString.trim() =="") null else x.toString.trim()))
      .map(r=>Row(r:_*))
    val schema = StructType(name.map(fieldName => StructField(fieldName, StringType, true)))
    hiveCtx.createDataFrame(data, schema)
  }

  def fieldDescribe(sc: SparkContext,hiveCtx: HiveContext,rawdbName:String,hivedbName:String,tableName:String,
                    dfMemo:DataFrame,insertTableName:String,coverThreshold:Double = -1.0) = {
    val df_raw = hiveCtx.sql(s"select * from $hivedbName.$tableName")
    val df_data = fieldProcessing(hiveCtx,df_raw)
    val dfSize: Double = df_data.count().toDouble
    //assert(dfSize>0)
    val columnName: Array[String] = df_data.columns
    val rawDescr: SchemaRDD = df_data.describe(columnName.toSeq:_*)

    // save DataFrame
    val fieldName: Array[String] = rawDescr.columns.drop(1)
    val dataTranspose: Array[Row] = fieldName
      .map(x =>(Array(rawdbName,hivedbName,tableName,x) ++
        rawDescr.select(x).map(x => x.toString.replaceAll("\\[|\\]", "")).collect()))
      .map(x=>Row(x:_*))

    // Trans to DataFrame
    val dfName: Array[String] =
      Array("rawdb_name","hivedb_name","table_name_r","field_name_r","count","mean","stddev","min","max")
    val data: RDD[Row] = sc.parallelize(dataTranspose)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescr: SchemaRDD = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)
    dfDescr.show()
    //  join data type Info
    val dfFieldType: SchemaRDD = hiveCtx.sql(s"desc $hivedbName.$tableName").select("col_name","data_type")
    val dfOutPut = dfDescr.join(dfFieldType,dfDescr("field_name_r") === dfFieldType("col_name") ,"left_outer")

    // join data memo Info
    val dfMemoName = dfMemo.columns
    val dfCurMemo = dfMemo.filter(s"${dfMemoName(0)} = '${tableName}'")
    val dfOutPut_v2 = dfOutPut.join(dfCurMemo,dfOutPut("field_name_r") === dfCurMemo(s"${dfMemoName(1)}"),"left_outer")
      .select("rawdb_name","hivedb_name","table_name_r","field_name_r","data_type",s"${dfMemoName(2)}","coverage_rate","count",
      "mean","stddev","min","max")
      .filter(s"coverage_rate > $coverThreshold")
    dfOutPut_v2.show()
    // insert into hive
    dfOutPut_v2.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    hiveCtx.sql(s"create table $insertTableName as select * from tmp_table ")

//    hiveCtx.sql(s"insert into table  $insertTableName select * from tmp_table ")
  }

  def file_name()={
    val all_file_name=
      List(
        "beehive_t_bee_customer"
//        "beehive_t_bee_marketing_info" ,
//        "beehive_t_bee_transport"
      )
    all_file_name.map(x=>x.trim().toLowerCase())
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("fieldDescribe")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")

    val tableList = file_name()
    hiveCtx.sql("use test")
    //val tableList: Array[String] = hiveCtx.sql("select * from test.useful_table_ecif").map(r=>r(0).toString).collect()
    //table memo info (tablename|fieldname|comment)
    val df_memo = hiveCtx.sql("select * from dw.table_field_desc")
    // tabledesctibe
    val tableDescribe = fieldDescribe(sc,hiveCtx,_:String,"ods",_:String,df_memo,"test.table_describe_clean_new",0.3)
    for (tableName <- tableList){

      val rawdbName = tableName.split("_")(0).toLowerCase()
      println(tableName,rawdbName)
      tableDescribe(rawdbName,tableName)
    }
  }
}
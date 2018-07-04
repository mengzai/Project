package cn.creditease.invest.test
/**
  * Created by Administrator on 2017/8/17.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

import scala.collection.Seq

object data_cleaning {

  def fieldDistinctCount(sc:SparkContext,hiveCtx:HiveContext,data_Df: DataFrame,processingNumber:Int = 100) = {
    val fieldName: Array[String] = data_Df.columns     //取数据的列名
    var processingField: Array[String] = fieldName  //作为中间变量
    var distinctCountOut: List[Long] = List()
    //  Iterative processing
    println("start1")

    while (processingField.length>processingNumber){
      val subField: Array[String] = processingField.slice(0,processingNumber) //每次取100个进行批量跑批覆盖率
      val distinctCount: Seq[Long] =data_Df.agg(countDistinct(subField(0)),subField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
        .first().toSeq.asInstanceOf[Seq[Long]]
      distinctCountOut = distinctCountOut ++ distinctCount //
      processingField = processingField.diff(subField)
    }
    println("start2")

    distinctCountOut = distinctCountOut ++ data_Df.agg(countDistinct(processingField(0)),processingField.drop(1).map(x=>countDistinct(x)).toSeq:_*)
      .first().toSeq.asInstanceOf[Seq[Long]]
    println("start3")
    val dataTranspose: Array[Row] =fieldName.zip(distinctCountOut).map(x=>Row(x._1,x._2))

    val dfName: Array[String] =  Array("field_name","distinct_value")
    val data: RDD[Row] = sc.parallelize(dataTranspose)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescrOut = hiveCtx.createDataFrame(data, schema)
    dfDescrOut.show()


    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    hiveCtx.sql(s"create table test.dainxiao_model_data_distinct  as select * from tmp_table ")
    dfDescrOut
  }


  def fieldProcessing(hiveCtx:HiveContext,df:DataFrame) = {
    val name: Array[String] = df.columns
    val data: RDD[Row] = df.map(r=>r.toSeq.map(x=>
      if(x == null) null else if (x.toString.trim() =="") null else x.toString.trim()))
      .map(r=>Row(r:_*))
    val schema = StructType(name.map(fieldName => StructField(fieldName, StringType, true)))
    hiveCtx.createDataFrame(data, schema)
  }

  def fieldDescribe(sc: SparkContext,hiveCtx: HiveContext,tableName:String) = {
    val df_raw = hiveCtx.sql(s"select * from $tableName")
    val df: SchemaRDD = fieldProcessing(hiveCtx,df_raw)
    val dfSize: Double = df.count().toDouble
    //assert(dfSize>0)
    val columnName: Array[String] = df.columns

    val rawDescrbe: SchemaRDD = df.describe(columnName.toSeq:_*)

    val fieldName_distinct: SchemaRDD =fieldDistinctCount(sc,hiveCtx,df)
    // save DataFrame
    val fieldName: Array[String] = rawDescrbe.columns.drop(1)
    val dataTranspose: Array[Row] = fieldName
      .map(x =>
        (Array(tableName,x) ++ fieldName_distinct.filter(s"field_name=$x").map(x=>x.get(2)).collect()
          ++ rawDescrbe.select(x).map(x => x.toString.replaceAll("\\[|\\]",""))
          .collect())
      ).map(x=>Row(x:_*))


    //     Trans to DataFrame
    val dfName: Array[String] =  Array("table_name","field_name","count","mean","stddev","min","max","distinct_value")
    val data: RDD[Row] = sc.parallelize(dataTranspose)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescrOut = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)
    dfDescrOut.show()




    // insert into hive
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    hiveCtx.sql(s"create table test.dainxiao_model_data_describe  as select * from tmp_table ")
  }

  def file_name()={

    val file_name=List(
      "dainxiao_model_data"
    )
    file_name
  }


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("fieldDescribe")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")

    val tableList = file_name()
    hiveCtx.sql(s"use test")

    val tableDescribe = fieldDescribe(sc,hiveCtx,_:String)
    for (tableName <- tableList){
      tableDescribe("test."+tableName)
    }
  }
}
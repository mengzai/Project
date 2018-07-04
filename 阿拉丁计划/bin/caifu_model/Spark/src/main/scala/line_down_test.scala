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

object line_down_test {
  def fieldDescribe(sc: SparkContext,hiveCtx: HiveContext,tableName:String) = {
    val df = hiveCtx.sql(s"select msn ,qq  from $tableName")
    val dfSize: Double = df.count().toDouble
    //assert(dfSize>0)
    val columnName: Array[String] = df.columns

    df.show()

    val rawDescrbe: SchemaRDD = df.describe(columnName.toSeq:_*)

    // save DataFrame
    val fieldName: Array[String] = rawDescrbe.columns.drop(1)
    val dataTranspose: Array[Row] = fieldName
      .map(x =>(Array(tableName,x) ++ rawDescrbe.select(x).map(x => x.toString.replaceAll("\\[|\\]","")).collect()))
      .map(x=>Row(x:_*))

//     Trans to DataFrame
    val dfName: Array[String] =  Array("table_name","field_name","count","mean","stddev","min","max")
    val data: RDD[Row] = sc.parallelize(dataTranspose)
    //DataFrame (field: Array[String],data: RDD[Row])
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val dfDescrOut = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)
    dfDescrOut.show()

    // insert into hive
    dfDescrOut.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    hiveCtx.sql(s"insert into table testtable_state_describe select * from tmp_table")
  }

  def file_name()={

    val file_name=List(
      "bs_tsm_user"
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
      tableDescribe("ods."+tableName)
    }
  }
}
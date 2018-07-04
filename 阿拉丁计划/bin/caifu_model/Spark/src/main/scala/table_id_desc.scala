package cn.creditease.invest.test
/**
  * Created by wq on 2017/9/4.
  */
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

object table_id_desc {
  def fieldDistinctCount(data: DataFrame,target_table:String,processingNumber:Int = 100) = {
    val fieldName: Array[String] = data.columns
    var processingField: Array[String] = fieldName
    processingField.foreach(println)
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
    fieldName.zip(distinctCountOut).map(x=>Row(target_table,x._1,x._2.toString))
  }

  def get_target_data(sc:SparkContext,hiveCtx:HiveContext,target_table: String ,target_table_field: Array[String] ,hivedbName:String) ={
    val df_raw = hiveCtx.sql(s"select * from $hivedbName.$target_table")
    val target_data=df_raw.select(target_table_field.map(c => col(c)): _*)
    val returen_data: Array[Row] =fieldDistinctCount(target_data,target_table)

    val dfName: Array[String] = Array("table_name","field_name","dsitinct_count")
    val data: RDD[Row] = sc.parallelize(returen_data)
    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
    val distinct_table: SchemaRDD = hiveCtx.createDataFrame(data, schema)
    distinct_table.show()
    //将数据写入到cleaning库中
    // insert into hive
    distinct_table.registerTempTable("tmp_table")
    hiveCtx.sql(s"use test")
    hiveCtx.sql(s"insert into table test.field_distinct_value_new   select * from tmp_table")
  }

  def file_name()={
    val all_file_name=
      List(
//        "beehive_t_bee_customer",
        "beehive_t_bee_marketing_info" ,
        "beehive_t_bee_transport"
      )
    all_file_name.map(x=>x.trim().toLowerCase())
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("fieldDescribe")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")

    hiveCtx.sql("use test")

    val hivedbName="test"
    val tableName="table_describe_clean_new"  //dw.table_describe 为覆盖率相关统计信息的相关表,且仅保存覆盖率>0.3
    val database_test="test"        //use test库
    val database_ods="ods"          //use ods库
    val is_discover="discover"      //是否是discover源库 discover源库数据在test库 其它表在ods中

    //得到有用表名称   dw.table_describe 为指定的有用表；有用字段及描述信息
//    val all_use_table_name: Array[String] = hiveCtx.sql(s"select distinct table_name  from $hivedbName.$tableName")
//      .map(r => r.getAs[String](0)).collect()

    val all_use_table_name: Seq[String] =file_name()

    //对每个有用表进行抽取相应有用字段
    for (target_table: String <- all_use_table_name){
      //抽取对应有用表的有用字段
      println(target_table)
      val target_table_filed: Array[String] = hiveCtx.sql(s"select distinct field_name  from $hivedbName.$tableName where table_name='$target_table' ")
        .map(r => r.getAs[String](0)).collect()
      target_table_filed.foreach(println)
      //将有用表 的有用字段对应抽取到cleaning库中
      if (target_table.toLowerCase.contains(is_discover.toLowerCase))

        get_target_data(sc,hiveCtx,target_table,target_table_filed,database_test)
      else get_target_data(sc,hiveCtx,target_table,target_table_filed,database_ods)
    }
   }
}
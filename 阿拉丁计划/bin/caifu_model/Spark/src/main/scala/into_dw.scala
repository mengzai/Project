package cn.creditease.invest.test
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HiveContext

/**
  * Created by wanglili on 17/8/29.
  */
object into_dw {

  def select_useful_table(hiveCtx:HiveContext,hivedbName:String,tableName:String)={
    val all_use_table_name: Array[String] = hiveCtx.sql(s"select distinct table_name  from $hivedbName.$tableName")
      .map(r => r.getAs[String](0)).collect()
    all_use_table_name
  }

  def select_target_table_filed(hiveCtx:HiveContext,target_table:String,hivedbName:String,tableName:String)={
    val target_table_filed: Array[String] = hiveCtx.sql(s"select field_name  from $hivedbName.$tableName where table_name='$target_table' ")
      .map(r => r.getAs[String](0)).collect()
    target_table_filed
  }

  def insert_to_dw(hiveCtx:HiveContext,tableName: String ,target_table_filed: Array[String] ,hivedbName:String) ={
    val df_raw = hiveCtx.sql(s"select * from $hivedbName.$tableName")
    val columnNames: Array[String] =df_raw.columns
    val taeget_data=df_raw.select(columnNames.map(c => col(c)): _*)

    taeget_data.printSchema()
    //将数据写入到cleaning库中
//    df_raw.registerTempTable("tmp_table")
//    hiveCtx.sql("use cleaning")
//    hiveCtx.sql(s"create table $tableName as select * from tmp_table")
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("into_dw")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")
    //得到有用表名称
    val all_use_table_name= select_useful_table(hiveCtx,"dw","table_describe")

    //对每个有用表进行抽取相应有用字段
    for (tableName: String <- all_use_table_name){
      //抽取对应有用表的有用字段，
      val target_table_filed: Array[String] =select_target_table_filed(hiveCtx,tableName,"dw","table_describe")
      //将有用表 的有用字段对应抽取到cleaning库中
      if (tableName.toLowerCase.contains("discover".toLowerCase)==true)
          insert_to_dw(hiveCtx,tableName,target_table_filed,"test")
      else insert_to_dw(hiveCtx,tableName,target_table_filed,"ods")
    }
  }
}

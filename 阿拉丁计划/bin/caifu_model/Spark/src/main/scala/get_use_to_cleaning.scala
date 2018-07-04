package cn.creditease.invest.test
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HiveContext

/**
  * Created by wanglili on 17/8/29.
  */
object get_use_to_cleaning {

  def insert_to_cleaning(hiveCtx:HiveContext,target_table: String ,target_table_field: Array[String] ,hivedbName:String) ={
    val df_raw = hiveCtx.sql(s"select * from $hivedbName.$target_table")
    val target_data=df_raw.select(target_table_field.map(c => col(c)): _*)
    target_data.printSchema()
    //将数据写入到cleaning库中

    target_data.registerTempTable("tmp_table")
    hiveCtx.sql("use cleaning")
    hiveCtx.sql(s"create table  if not EXISTS  $target_table as select * from tmp_table ")
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("get_use_to_cleaning")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR"  )

    val hivedbName="dw"
    val tableName="table_describe_clean"  //dw.table_describe 为覆盖率相关统计信息的相关表,且仅保存覆盖率>0.3
    val database_test="test"        //use test库
    val database_ods="ods"          //use ods库
    val is_discover="discover"      //是否是discover源库 discover源库数据在test库 其它表在ods中

    //得到有用表名称   dw.table_describe 为指定的有用表；有用字段及描述信息
    val all_use_table_name: Array[String] = hiveCtx.sql(s"select distinct table_name  from $hivedbName.$tableName")
      .map(r => r.getAs[String](0)).collect()

//    all_use_table_name.map(x=>
//    val target_table_filed: Array[String] = hiveCtx.sql(s"select distinct field_name  from $hivedbName.$tableName where table_name='$x' ")
//      .map(r => r.getAs[String](0)).collect()

//    )

    //对每个有用表进行抽取相应有用字段
    for (target_table: String <- all_use_table_name){
      //抽取对应有用表的有用字段
      println(target_table)
      val target_table_filed: Array[String] = hiveCtx.sql(s"select distinct field_name  from $hivedbName.$tableName where table_name='$target_table' ")
        .map(r => r.getAs[String](0)).collect()

      //将有用表 的有用字段对应抽取到cleaning库中
      if (target_table.toLowerCase.contains(is_discover.toLowerCase))
        insert_to_cleaning(hiveCtx,target_table,target_table_filed,database_test)
      else insert_to_cleaning(hiveCtx,target_table,target_table_filed,database_ods)
    }
  }
}
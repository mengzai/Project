package cn.creditease.invest.test
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by gangfang6 on 2017/7/14.
  */
object TestHive {
  case class Person(name:String,age:Int)
  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setAppName("test_hive").setMaster("local")
    val sc=new SparkContext(conf)
    val sqlCtx = new SQLContext(sc)
    val path="./data/people.txt"
    val df=sqlCtx.read.json(path)
    print(df.show())
    print(df.printSchema())

//    df.select("name").show()
//    df.select("age").show()
//    df.select(df("name"),df("age")+1).show()
//    df.filter(df("age")>21).show()
//    df.groupBy(df("age")).count().show()
//
    import sqlCtx.implicits._

//    val rdd=sc.textFile(path).map(_.split(","))
//    val rddContainingCaseClass=rdd.map(p=>Person(p(0),p(1).trim.toInt))
//    val people=rddContainingCaseClass.toDF()
//    people.show()
    val fieldName: Array[Int] =Array(1,2,3,4)


  }
}

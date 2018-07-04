import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wanglili on 17/7/18.
  */
class local_test_hive {
  //定义表结构
  case class Person(name: String, age:Int)

  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setAppName("locla_test_hive").setMaster("local")
    val sc=new SparkContext(conf)
    val sqlCtx=new SQLContext(sc)

    val path="./data/people.txt"
    val df=sqlCtx.read.json(path)

    print (df.show())
  }
}

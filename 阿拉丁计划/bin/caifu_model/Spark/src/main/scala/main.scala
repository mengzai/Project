package cn.creditease.invest.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.hive.HiveContext
// import scala.util.matching.Regex
import org.apache.spark._

import scala.math.random
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType}
import org.apache.spark.sql.functions._

/**
  *
  */
object main {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("sdk_input_log_feature")
      .setSparkHome(System.getenv("SPARK_HOME"))
    val sc = new SparkContext(conf)
    conf.getAll.foreach(println)
    println("----" * 5)

    sc.setLogLevel("WARN")
    sc.getConf.getAll.foreach(println)
    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    // Logger.getLogger("akka").setLevel(Level.ERROR)

    val hiveContext = new HiveContext(sc)
    // hiveContext.sql("use test")
    //    val dbs = hiveContext.sql("show databases")
    //    dbs.foreach(println)
    hiveContext.sql("use test")
    val tables = hiveContext.sql("show tables")
    tables.printSchema()

    val slices = if (args.length > 0) args(0).toInt else 2
    val n = math.min(100000L * slices, Int.MaxValue).toInt // avoid overflow
    val count = sc.parallelize(1 until n, slices).map { i =>
      val x = random * 2 - 1
      val y = random * 2 - 1
      if (x*x + y*y < 1) 1 else 0
    }.reduce(_ + _)
    println("Pi slices " + slices)
    println("Pi count " + count)
    println("Pi n " + n)
    println("Pi is roughly " + 4.0 * count / n)

    sc.stop()

  }
}
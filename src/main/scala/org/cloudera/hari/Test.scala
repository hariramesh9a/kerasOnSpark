package org.cloudera.hari

import org.apache.spark.{SparkConf, SparkContext}

object Test {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("basics").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val dataDir = "/Users/harumughan/IdeaProjects/DeepLearn/data/"

    val suppRDD = sc.textFile(dataDir + "supplier.dat")

   suppRDD.take(1)

  }

}

package org.cloudera.hari

import org.apache.spark.sql.{Dataset, SparkSession}
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.factory.Nd4j

object KerasServingSpark {


  def main(args: Array[String]): Unit = {
    // load the model
    val spark = SparkSession
      .builder()
      .appName("Spark SQL Keras Serving").master("local[*]")
      .getOrCreate()


    val modelPath = getClass.getResource("/model.h5")
    print(modelPath)
    val model = KerasModelImport.
      importKerasSequentialModelAndWeights(modelPath.getPath)


    // make a random sample


    import spark.implicits._

    val df = Seq((1, 2, 4), (3, 5, 6)).toDF("f1", "f2", "f3")
    df.show()
    val nd = Nd4j.create(Array[Float](1, 2, 3, 4, 5, 6), Array[Int](2, 3))
    val count = df.count().toLong
    val arr = df.collect.flatMap(_.toSeq).map(_.toString.toDouble)
    val ndarray = Nd4j.create(Array[Float](1, 2, 3, 4, 5, 6), Array[Int](2, 3))

    val ndarray1 = Nd4j.create(arr, Array(count, 3))

    // get the prediction
    val prediction = model.output(ndarray1)
    print(prediction)


  }
}

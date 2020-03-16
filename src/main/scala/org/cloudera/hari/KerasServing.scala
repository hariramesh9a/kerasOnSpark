package org.cloudera.hari

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File

object KerasServing {


  def main(args: Array[String]): Unit = {


    // load the model


    val modelPath = getClass.getResource("/model.h5")
    print(modelPath)
    val model = KerasModelImport.
      importKerasSequentialModelAndWeights(modelPath.getPath)


    // make a random sample


    val nd = Nd4j.create(Array[Float](1, 2, 3, 4, 5, 6), Array[Int](2, 3))
    println(nd)
    // get the prediction
    val prediction = model.output(nd)
    print(prediction)
  }
}

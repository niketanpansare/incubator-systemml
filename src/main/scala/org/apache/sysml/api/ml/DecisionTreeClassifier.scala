/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api.ml

import org.apache.spark.rdd.RDD
import java.io.File
import org.apache.spark.SparkContext
import org.apache.spark.ml.{ Estimator, Model }
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.param.{ DoubleParam, Param, ParamMap, Params }
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.runtime.DMLRuntimeException
import org.apache.sysml.runtime.instructions.spark.utils.{ RDDConverterUtilsExt => RDDConverterUtils }
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._

object DecisionTreeClassifier {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "decision-tree.dml"
}

/**
  * DecisionTreeClassifierModel Scala API
  */
class DecisionTreeClassifier(override val uid: String, val sc: SparkContext)
    extends Estimator[DecisionTreeClassifierModel]
    with HasBins
    with HasDepth
    with HasNumLeaf
    with HasNumSamples
    with HasImpurity
    with BaseSystemMLClassifier {
  
  def setBins(value: Long)  = set(bins, value.toInt)
  def setDepth(value: Long) = set(depth, value.toInt)
  def setNumLeaf(value: Long) = set(numLeaf, value.toInt)
  def setNumSamples(value: Long)  = set(numSamples, value.toInt)
  def setBins(value: Int)  = set(bins, value)
  def setDepth(value: Int) = set(depth, value)
  def setNumLeaf(value: Int) = set(numLeaf, value)
  def setNumSamples(value: Int)  = set(numSamples, value)
  def setImpurity(value: String) = {
    if(value.equalsIgnoreCase("gini")) set(impurity, "Gini")
    else if(value.equalsIgnoreCase("entropy")) set(impurity, "entropy")
    else {
      throw new RuntimeException("Unsupported impurity: " + value + ". Supported values are Gini, entropy")
    }
  }

  override def copy(extra: ParamMap): DecisionTreeClassifier = {
    val that = new DecisionTreeClassifier(uid, sc)
    copyValues(that, extra)
  }
  
  def fit(X_file: String, y_file: String): DecisionTreeClassifierModel = {
    mloutput = baseFit(X_file, y_file, sc)
    new DecisionTreeClassifierModel(this)
  }

  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): DecisionTreeClassifierModel = {
    mloutput = baseFit(X_mb, y_mb, sc)
    new DecisionTreeClassifierModel(this)
  }

  def fit(df: ScriptsUtils.SparkDataType): DecisionTreeClassifierModel = {
    mloutput = baseFit(df, sc)
    new DecisionTreeClassifierModel(this)
  }
  
  var _rMB: MatrixBlock = null
  def setR(rMB: MatrixBlock) {
    _rMB = rMB
  }

  def getTrainingScript(isSingleNode: Boolean): (Script, String, String) = {
    var script = dml(ScriptsUtils.getDMLScript(DecisionTreeClassifierModel.scriptPath))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$M", " ")
      .in("$bins", toDouble(getBins))
      .in("$depth", toDouble(getDepth))
      .in("$num_leaf", toDouble(getNumLeaf))
      .in("$num_samples", toDouble(getNumSamples))
      .in("$impurity", getImpurity)
      .out("M")
    if(_rMB != null) {
      script = script.in("$R", "ignore").in("R", _rMB)
    }
    (script, "X", "Y_bin")
  }
}

object DecisionTreeClassifierModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "decision-tree-predict.dml"
}

/**
  * Decision tree classifier Scala API
  */
class DecisionTreeClassifierModel(override val uid: String)(estimator: DecisionTreeClassifier, val sc: SparkContext, val rMB: MatrixBlock)
    extends Model[DecisionTreeClassifierModel]
    with HasBins
    with HasDepth
    with HasNumLeaf
    with HasNumSamples
    with HasImpurity
    with BaseSystemMLClassifierModel {
  override def copy(extra: ParamMap): DecisionTreeClassifierModel = {
    val that = new DecisionTreeClassifierModel(uid)(estimator, sc, rMB)
    copyValues(that, extra)
  }
  var outputRawPredictions                               = true
  def setOutputRawPredictions(outRawPred: Boolean): Unit = outputRawPredictions = outRawPred
  def this(estimator: DecisionTreeClassifier) = {
    this("model")(estimator, estimator.sc, estimator._rMB)
  }
  def getPredictionScript(isSingleNode: Boolean): (Script, String) = {
    var script = dml(ScriptsUtils.getDMLScript(DecisionTreeClassifierModel.scriptPath))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$P", " ")
      .in("$M", " ")
      .out("Y_predicted")
    if(rMB != null) {
      script = script.in("$R", "ignore").in("R", rMB)
    }

    val w    = estimator.mloutput.getMatrix("M")
    val wVar = "M"

    val ret = if (isSingleNode) {
      script.in(wVar, w.toMatrixBlock, w.getMatrixMetadata)
    } else {
      script.in(wVar, w)
    }
    (ret, "X_test")
  } 
  
  override def allowsTransformProbability() = false

  def baseEstimator(): BaseSystemMLEstimator = estimator
  def modelVariables(): List[String]         = List[String]("B_out")

  def transform(X: MatrixBlock): MatrixBlock               = baseTransform(X, sc, "means")
  def transform(X: String): String                         = baseTransform(X, sc, "means")
  def transform_probability(X: MatrixBlock): MatrixBlock   = baseTransformProbability(X, sc, "means")
  def transform_probability(X: String): String             = baseTransformProbability(X, sc, "means")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, sc, "means")
}

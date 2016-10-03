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
package org.apache.sysml.api.dl

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.api.ml.ScriptsUtils
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import scala.collection.JavaConversions._
import java.util.ArrayList
import caffe.Caffe.Phase
import caffe.Caffe

object CaffeModel  {
  // For Testing:
  // org.apache.sysml.api.dl.CaffeModel 
  def main(args: Array[String]): Unit = {
    if(args.length == 1) {
      val model = new CaffeModel(args(0), 1, 256, 256)
      val net = new CaffeNetwork(model.solver.getNet, Caffe.Phase.TRAIN, 1, 256, 256)
      model.train(model.solver, net)
    }
    else
      throw new LanguageException("Expected input solver file")
  }
}
class CaffeModel(val solverFilePath:String, val numChannels:Int, val inputHeight:Int, val inputWidth:Int) {
	val prefix = "nn";
	val layerDir = prefix + "/layers/";
	val optimDir = prefix + "/optim/";
			
	// val solverMapping = Map("sgd" -> "sgd_momentum", "adagrad" -> "adagrad", "adam" -> "adam", "nesterov" -> "sgd_nesterov", "rmsprop" -> "rmsprop")
	
	var currentPhase = Caffe.Phase.TRAIN
	var solver = Utils.readCaffeSolver(solverFilePath)
	
	
	// --------------------------------------------------------------
	// External APIs
	// Allows for reloading of the solver and network
//	def load(solverFilePath:String, numChannels:Int, inputHeight:Int, inputWidth:Int) = {
//	  solver = Utils.readCaffeSolver(solverFilePath)
//	  net = CaffeNetwork(solver.getNet(), currentPhase, numChannels, inputHeight, inputWidth)
//	}
	def fit(df: ScriptsUtils.SparkDataType):CaffeModel = {
	  return this
	}
	def fit(X_mb: MatrixBlock, y_mb: MatrixBlock):CaffeModel = {
	  return this
	}
	def transform(X: MatrixBlock): MatrixBlock = {
	  null
	}
	def transform(df: ScriptsUtils.SparkDataType): ScriptsUtils.SparkDataType = {
	  null
	}
	
	// --------------------------------------------------------------
	// Script generator
	def train(solver:SolverParameter, net:CaffeNetwork):String = {
	  currentPhase = Caffe.Phase.TRAIN
	  
	  val dmlScript = new StringBuilder
	  for(layers <- net.getLayers) {
	    net.getCaffeLayer(layers).init(dmlScript)
	  }
	  for(layers <- net.getLayers) {
	    net.getCaffeLayer(layers).forward(dmlScript)
	  }
	  for(layers <- net.getLayers) {
	    net.getCaffeLayer(layers).backward(dmlScript)
	  }
	  System.out.println(dmlScript.toString())
	  null
	}
	
	def getPrevLayerIDForSingleInputLayer(prevLayerIDs:List[Int]): Int = {
	  if(prevLayerIDs.size != 1) {
	    throw new LanguageException("Multiple bottom layers is not handled")
	  }
	  prevLayerIDs.get(0)
	}
	

	// --------------------------------------------------------------
	// DML generation utility functions
	var prevTab:String = null
	var prevNumTabs = -1
	var numTabs = 2
	def tabs():String = if(numTabs == prevNumTabs) prevTab else { prevTab = ""; for(i <- 0 until numTabs) { prevTab += "\t" } ; prevTab  }
	def source(dir:String, file:String) =  "source(\"" + dir + file + ".dml" + "\") as " + file + "\n"
	// --------------------------------------------------------------

	
}
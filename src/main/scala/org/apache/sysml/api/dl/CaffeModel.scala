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

object CaffeModel  {
  def main(args: Array[String]): Unit = {
    if(args.length == 1) {
      val model = new CaffeModel(args(0))
      model.train(model.solver, model.net)
    }
    else
      throw new LanguageException("Expected input solver file")
  }
}
class CaffeModel(val solverFilePath:String) {
	val prefix = "nn";
	val layerDir = prefix + "/layers/";
	val optimDir = prefix + "/optim/";

	// Handled differently Data, Accuracy
	val layerMapping = Map("convolution" -> ("conv_builtin", 3), "pooling" -> ("max_pool_builtin", 3), "innerproduct" -> ("affine", 1), 
			"relu" -> ("relu", 1), "softmaxwithloss" -> ("cross_entropy_loss", 1), "dropout" -> ("dropout", 2))
	
			
	val solverMapping = Map("sgd" -> "sgd_momentum", "adagrad" -> "adagrad", "adam" -> "adam", "nesterov" -> "sgd_nesterov", "rmsprop" -> "rmsprop")
	
	var solver = Utils.readCaffeSolver(solverFilePath)
	var net = Utils.readCaffeNet(solver)
	
	// --------------------------------------------------------------
	// External APIs
	// Allows for reloading of the solver and network
	def load(solverFilePath:String) = {
	  solver = Utils.readCaffeSolver(solverFilePath)
	  net = Utils.readCaffeNet(solver)
	}
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
	
	// Layer name, Layer parameter, layerID
	def getLayers(net:NetParameter):Map[String, (LayerParameter, Int)] = {
	  var layerID:Int = 1
	  val ret = Map[String, (LayerParameter, Int)]()
	  for(l <- net.getLayerList) {
	    layerID = layerID + 1
	    ret.put(l.getName, (l, layerID))
	  }
	  ret
	}
	
	
	def train(solver:SolverParameter, net:NetParameter):String = {
	  val dmlScript = new StringBuilder
	  dmlScript.append(getImportLayers(net)) 
	  dmlScript.append(source(optimDir, getOrThrowException(solverMapping.get(solver.getType.toLowerCase), solver.getType.toLowerCase)))
	  
	  // [Layer name, (Layer parameter, layerID)]
	  val layers:Map[String, (LayerParameter, Int)] = getLayers(net)
	  val sortedLayers:ArrayList[String] = Utils.getTopologicalSortedLayers(net, layers)
	  
	  // TODO: Support passing of bias fillers in convolution init: See SYSTEMML-1000
	  dmlScript.append("# TODO: Init")
	  
	  numTabs = 2
	  dmlScript.append("# Compute forward pass\n")
	  for(i <- 0 until sortedLayers.length) {
	    val layerName = sortedLayers(i)
	    val tmp = layers.getOrElse(layerName, null)
	    val layer = tmp._1
	    val layerID = tmp._2
	    val numOut = getNumOut(layer.getType)
	    forward(layer, layerName, layerID, numOut, getPreviousLayerID(), dmlScript)
	  }
	  
	  dmlScript.append("# TODO: backward")
	  System.out.println(dmlScript.toString())
	  null
	}
	

	def forward(layer:LayerParameter, layerName:String, layerID:Int, numOut:Int, prevLayerID:Int, dmlScript:StringBuilder): Unit = {
	  dmlScript.append(tabs)
	  // Return arguments
	  numOut match {
	    case 1 => dmlScript.append(outputVar(layerID))
	    case 2 => {
	      layer.getType.toLowerCase match {
	        case "dropout" => dmlScript.append("[" + getCommaSepList(outputVar(layerID), maskVar(layerID)) + "]")
	        case _ => throw new LanguageException("Number of outputs: " + numOut + " is not supported for " + layer.getType)
	      }
	    }
	    case 3 => dmlScript.append("[" + getCommaSepList(outputVar(layerID), outputHeightVar(layerID), outputWidthVar(layerID)) + "]")
	    case _ => throw new LanguageException("Number of outputs: " + numOut + " is not supported.")
	  }
	  
	  dmlScript.append(" = " + layerName + "::forward(")
	  
	  // Arguments
	  layer.getType.toLowerCase match {
	    case "convolution" => {
	      val convParams = getConvolutionParams(layer.getConvolutionParam)
	      dmlScript.append(getCommaSepList(outputVar(prevLayerID), weightVar(layerID), biasVar(layerID), 
	        outputChannelVar(prevLayerID), outputHeightVar(prevLayerID), outputWidthVar(prevLayerID),
	        // (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, numFilters)
	        convParams._1.toString, convParams._2.toString, convParams._3.toString, convParams._4.toString, convParams._5.toString, convParams._6.toString
	        ) + ")\n") 
	      appendDimensions(dmlScript, outputChannelVar(layerID), convParams._7.toString) // output #channels = #filter
	    }   
	    case "pooling" =>  {
	      val poolingParam = getPoolingParams(layer.getPoolingParam)
	      dmlScript.append(getCommaSepList(outputVar(prevLayerID),
	          outputChannelVar(prevLayerID), outputHeightVar(prevLayerID), outputWidthVar(prevLayerID),
	          // (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
	          poolingParam._1.toString, poolingParam._2.toString, poolingParam._3.toString, poolingParam._4.toString
	          // TODO: Support passing of padding: See SYSTEMML-1001
	          // , poolingParam._5.toString, poolingParam._6.toString
	        ) + ")\n")
	        appendDimensions(dmlScript, outputChannelVar(layerID), outputChannelVar(prevLayerID)) // output #channels = input #channels
	    }
	    case "innerproduct" =>  {
	      dmlScript.append(getCommaSepList(outputVar(prevLayerID), weightVar(layerID), biasVar(layerID)) + ")\n")
	      // output tensor of shape = [input number of images, innerProduct.getNumOutput, 1, 1]
	      appendDimensions(dmlScript, outputChannelVar(layerID), layer.getInnerProductParam.getNumOutput.toString(),
	          outputHeightVar(layerID), "1", outputWidthVar(layerID), "1")
	    }
	    case "relu" => {
	      dmlScript.append(outputVar(prevLayerID) + ")\n")
	      appendSameChannelHW(dmlScript, layerID, prevLayerID)
	    }
	    case "softmaxwithloss" =>  {
	      // TODO:
	      dmlScript.append("... TODO ..."+ ")\n")
	    }
	    case "dropout" =>  {
	      dmlScript.append(outputVar(prevLayerID) + ")\n")
	      appendSameChannelHW(dmlScript, layerID, prevLayerID)
	    }
	  }
	}
	
	def appendSameChannelHW(dmlScript:StringBuilder, layerID:Int, prevLayerID:Int):Unit = {
	  appendDimensions(dmlScript, outputChannelVar(layerID), outputChannelVar(prevLayerID), outputHeightVar(layerID), outputHeightVar(prevLayerID),
	          outputWidthVar(layerID), outputWidthVar(prevLayerID))
	}
	
	// TODO:
	def getPreviousLayerID():Int = 0
	def appendDimensions(dmlScript:StringBuilder, args:String*) = {
	  dmlScript.append(tabs)
	  for(i <- 0 until (args.length/2)) {
	    dmlScript.append(args(i) + " = " + args(i+1) + "; ")
	  }
	  dmlScript.append("\n")
	}
	def outputVar(layerID:Int) = "out_" + layerID
	def weightVar(layerID:Int) = "W_" + layerID
	def maskVar(layerID:Int) = "mask_" + layerID
	def biasVar(layerID:Int) = "b_" + layerID
	def outputChannelVar(layerID:Int) = "C_" + layerID
	def outputHeightVar(layerID:Int) = "Hout_" + layerID
	def outputWidthVar(layerID:Int) = "Wout_" + layerID
	def getCommaSepList(arr:String*) = {
	  if(arr.length == 1) arr(0) else {
	    var ret = arr(0)
	    for(i <- 1 until arr.length) {
	      ret = ret + "," + arr(i)
	    }
	    ret
	  }
	}
	// Returns (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
	def getPoolingParams(poolingParam:caffe.Caffe.PoolingParameter) = {
	  if(poolingParam.getPool() != caffe.Caffe.PoolingParameter.PoolMethod.MAX)
	    throw new LanguageException("Only maxpooling is supported")
	  val kernel_h = if(poolingParam.hasKernelH()) poolingParam.getKernelH() else poolingParam.getKernelSize()
	  val kernel_w = if(poolingParam.hasKernelW()) poolingParam.getKernelW() else poolingParam.getKernelSize()
	  val stride_h = if(poolingParam.hasStrideH()) poolingParam.getStrideH() else poolingParam.getStride()
	  val stride_w = if(poolingParam.hasStrideW()) poolingParam.getStrideW() else poolingParam.getStride()
	  val pad_h = if(poolingParam.hasPadH()) poolingParam.getPadH() else poolingParam.getPad()
	  val pad_w = if(poolingParam.hasPadW()) poolingParam.getPadW() else poolingParam.getPad()
	  (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
	}
	// Returns (kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, numFilters)
	def getConvolutionParams(convParam: caffe.Caffe.ConvolutionParameter): (Int, Int, Int, Int, Int, Int, Int) = {
	  var stride_h = 1
	  var stride_w = 1
	  var pad_h = 0
	  var pad_w = 0
	  if(convParam.hasPadH())
			pad_h = convParam.getPadH();
		else if(convParam.getPadCount() > 0)
			pad_h = convParam.getPad(0);
		if(convParam.hasPadW())
			pad_w = convParam.getPadW();
		else if(convParam.getPadCount() > 0)
			pad_w = convParam.getPad(0);
		
		if(convParam.hasStrideH())
			stride_h = convParam.getStrideH();
		else if(convParam.getStrideCount() > 0)
			stride_h = convParam.getStride(0);
		if(convParam.hasStrideW())
			stride_w = convParam.getStrideW();
		else if(convParam.getStrideCount() > 0)
			stride_w = convParam.getStride(0);
		
		val kernel_h = if(convParam.hasKernelH()) convParam.getKernelH() else if(convParam.getKernelSizeCount() > 0) convParam.getKernelSize(0) else -1
		val kernel_w = if(convParam.hasKernelW()) convParam.getKernelW() else if(convParam.getKernelSizeCount() > 0) convParam.getKernelSize(0) else -1
		if(kernel_h == -1 || kernel_w == -1)
		  throw new LanguageException("Kernel height/width not available for convolution")
		(kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, convParam.getNumOutput)
	}

	// --------------------------------------------------------------
	// DML generation utility functions
	var prevTab:String = null
	var prevNumTabs = -1
	var numTabs = 2
	def tabs():String = if(numTabs == prevNumTabs) prevTab else { prevTab = ""; for(i <- 0 until numTabs) { prevTab += "\t" } ; prevTab  }
	def source(dir:String, file:String) =  "source(\"" + dir + file + ".dml" + "\") as " + file + "\n"
	// Iterates over caffe layers and generates source statements
	def getImportLayers(net:NetParameter) = net.getLayerList
	  .filter(l => l.getType.toLowerCase.compareTo("data") != 0)
	  .map(l => getOrThrowException1(layerMapping.get(l.getType.toLowerCase), l.getType.toLowerCase)).distinct.map(source(layerDir, _))
	
	def extractFromOption(opt:Option[(LayerParameter, Int)]):(LayerParameter, Int) = {
	  opt match {
	    case Some(value) => value
	    case None => throw new LanguageException("Expected option to be not null.")
	  }
	}
	def getOrThrowException1(opt:Option[(String, Int)], key:String):String = {
	  opt match {
	    case Some(value) => value._1
	    case None => throw new LanguageException(key + " is not yet supported.")
	  }
	}
	def getOrThrowException(opt:Option[String], key:String):String = {
	  opt match {
	    case Some(value) => value
	    case None => throw new LanguageException(key + " is not yet supported.")
	  }
	}
	def getNumOut(layerType:String): Int = {
	  layerMapping.get(layerType) match {
      case Some(val1) => val1._2
      case None => throw new LanguageException("Layer not found")
    }
	}
	// --------------------------------------------------------------

	
}
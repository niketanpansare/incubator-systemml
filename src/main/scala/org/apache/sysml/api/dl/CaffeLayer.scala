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

import caffe.Caffe.LayerParameter
import scala.collection.JavaConversions._
import org.apache.sysml.parser.LanguageException
import java.util.HashSet
import java.io.File
import org.apache.sysml.api.DMLScript
import org.apache.sysml.runtime.util.ConvolutionUtils

trait CaffeLayer {
  // -------------------------------------------------
  // Any layer that wants to reuse SystemML-NN has to override following methods that help in generating the DML for the given layer:
  def sourceFileName:String;
  def init(dmlScript:StringBuilder):Unit;
  def forward(dmlScript:StringBuilder, prefix:String):Unit;
  def backward(dmlScript:StringBuilder):Unit;
  def outputShape:(String, String, String) = bottomLayer.outputShape
  def weight():String = null;
  def bias():String = null;
  def dW():String = null;
  def dB():String = null;
  def computeLoss(dmlScript:StringBuilder, prefix:String):Unit = {}
  // -------------------------------------------------
  def source(dmlScript:StringBuilder):Unit = {
    CaffeClassifier.source(dmlScript, sourceFileName, CaffeClassifier.layerDir)
  }
  def bottomLayer:CaffeLayer = {
    val bottomLayerNames = net.getBottomLayers(param.getName)
    if(bottomLayerNames == null) throw new LanguageException("Expected atleast 1 bottom layer for " + param.getName)
    val ret = bottomLayerNames.map(l => net.getCaffeLayer(l)).toList
    if(ret.size != 1) throw new LanguageException("Expected only 1 bottom layer for " + param.getName)
    else ret(0)
  }
  def topLayer:CaffeLayer = {
    val ret = net.getTopLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    if(ret.size != 1) throw new LanguageException("Expected only 1 top layer")
    else ret(0)
  }
  def commaSep(arr:String*):String = {
	  if(arr.length == 1) arr(0) else {
	    var ret = arr(0)
	    for(i <- 1 until arr.length) {
	      ret = ret + "," + arr(i)
	    }
	    ret
	  }
	}
  def int_mult(v1:String, v2:String, v3:String):String = try { (v1.toDouble * v2.toDouble * v3.toDouble).toInt.toString } catch { case _:Throwable => "("+v1+"*"+v2+"*"+v3+")"}
  def addTab(dmlScript:StringBuilder):StringBuilder = { (0 until CaffeClassifier.numTabs).map(dmlScript.append("\t")); dmlScript }
  def param:LayerParameter
  def id:Int
  def net:CaffeNetwork
  def namespace:String = sourceFileName + "::"
  def outVar = "out" + id
  def dOut = "dOut" + id
  
  def isNumber(x: String):Boolean = x forall Character.isDigit
  def transpose(x:String):String = "t(" + x + ")"
}

class Data(val param:LayerParameter, val id:Int, val net:CaffeNetwork, val numChannels:Int, val inputHeight:Int, val inputWidth:Int) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = {
    if(param.hasTransformParam && param.getTransformParam.hasScale) {
      dmlScript.append("X_full = X_full * " + param.getTransformParam.getScale + "\n")
    }
    dmlScript.append("BATCH_SIZE = " + batchSize + "\n")
  }
  
  override def forward(dmlScript:StringBuilder, prefix:String) = { }
  override def outVar = "Xb"
  override def backward(dmlScript:StringBuilder) = { }
  override def outputShape = (numChannels.toString, inputHeight.toString, inputWidth.toString)
  // -------------------------------------------------
  val batchSize = param.getDataParam.getBatchSize
}

class SoftmaxWithLoss(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def source(dmlScript:StringBuilder):Unit = {
    if(!CaffeClassifier.alreadyImported.contains("softmax")) CaffeClassifier.source(dmlScript, "softmax", CaffeClassifier.layerDir)
    if(!CaffeClassifier.alreadyImported.contains("cross_entropy_loss")) CaffeClassifier.source(dmlScript, "cross_entropy_loss", CaffeClassifier.layerDir)
    CaffeClassifier.alreadyImported.add("softmax")
    CaffeClassifier.alreadyImported.add("cross_entropy_loss")
  }
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, prefix:String) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + bottomLayer.outVar + ")\n")
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "dProbs = cross_entropy_loss::backward(" + commaSep(outVar, "yb") + ")\n\t" + 
      dOut + " = " + namespace + "backward(" + commaSep("dProbs", bottomLayer.outVar) + ")\n")
  override def computeLoss(dmlScript:StringBuilder, prefix:String) = {
    dmlScript.append(prefix).append("tmp_loss = cross_entropy_loss::forward(" + commaSep(outVar, "yb") + ")\n")
    dmlScript.append(prefix).append("loss = loss + tmp_loss\n")
  }
  // -------------------------------------------------
}

class ReLU(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "relu"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder, prefix:String) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + bottomLayer.outVar + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayer.dOut, bottomLayer.outVar) + ")\n")
  // -------------------------------------------------
}

class Dropout(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "dropout"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder, prefix:String) = dmlScript.append(prefix).append(
      "[" + commaSep(outVar, maskVar) + "] = " + namespace + "forward(" + commaSep(bottomLayer.outVar, p, seed) + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayer.dOut, bottomLayer.outVar, p, maskVar) + ")\n")
  // -------------------------------------------------
  def maskVar = "mask" + id
  def p = param.getDropoutParam.getDropoutRatio.toString
  def seed = "-1"
}

class InnerProduct(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "affine"
  override def init(dmlScript:StringBuilder) = {
    dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + "init(" + commaSep(
          int_mult(bottomLayer.outputShape._1, bottomLayer.outputShape._2, bottomLayer.outputShape._3), numNeurons) + ")\n")
  }
  override def forward(dmlScript:StringBuilder, prefix:String) = dmlScript.append(prefix).append(
      outVar + " = " + namespace + "forward(" + commaSep(bottomLayer.outVar, weight, bias) + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace + 
      "backward(" + commaSep(topLayer.dOut, bottomLayer.outVar, weight, bias) + ")\n")
  
      
  // -------------------------------------------------
  def numNeurons = param.getInnerProductParam.getNumOutput.toString
  override def outputShape = ( param.getInnerProductParam.getNumOutput.toString, "1", "1" )
  override def weight = "W" + id
  override def bias = "b" + id
  override def dW = "dW" + id
  override def dB = "db" + id
}

class MaxPooling(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "max_pool_builtin"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder, prefix:String) = {
    val out2:String = if(isNumber(outputShape._2)) "ignore1_"+id else outputShape._2
    val out3:String = if(isNumber(outputShape._3)) "ignore2_"+id else outputShape._3
    dmlScript.append(prefix).append(
      "[" + commaSep(outVar, out2, out3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayer.outVar, numChannels,  bottomLayer.outputShape._2, bottomLayer.outputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) + ")\n")
  }
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayer.dOut, outputShape._2, outputShape._3, bottomLayer.outVar, 
                  numChannels, bottomLayer.outputShape._2, bottomLayer.outputShape._3, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)+ ")\n")
  override def outputShape = ( numChannels, outputHeight, outputWidth )
  // -------------------------------------------------
  def poolingParam = param.getPoolingParam
  def numChannels = bottomLayer.outputShape._1
  def kernel_h = if(poolingParam.hasKernelH) poolingParam.getKernelH.toString 
                   else poolingParam.getKernelSize.toString 
  def kernel_w = if(poolingParam.hasKernelW) poolingParam.getKernelW.toString 
                   else poolingParam.getKernelSize.toString
  def stride_h = if(poolingParam.hasStrideH) poolingParam.getStrideH.toString 
                   else poolingParam.getStride.toString
  def stride_w = if(poolingParam.hasStrideW) poolingParam.getStrideW.toString 
                   else poolingParam.getStride.toString
  def pad_h =   if(poolingParam.hasPadH) poolingParam.getPadH.toString 
                   else poolingParam.getPad.toString
  def pad_w =   if(poolingParam.hasPadW) poolingParam.getPadW.toString 
                   else poolingParam.getPad.toString
  val outputHeight =  try { ConvolutionUtils.getP(bottomLayer.outputShape._2.toLong, kernel_h.toLong, stride_h.toLong, pad_h.toLong).toString } catch { case _ : Throwable => "Hout" + id } 
  val outputWidth =  try { ConvolutionUtils.getQ(bottomLayer.outputShape._3.toLong, kernel_w.toLong, stride_w.toLong, pad_w.toLong).toString } catch { case _ : Throwable => "Wout" + id }
}

class Convolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "conv_builtin";
  override def init(dmlScript:StringBuilder) = { 
    val C = numChannels
    dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + 
      "init(" + commaSep(numKernels, C, kernel_h, kernel_w) + ")\n")
  }
  
  override def forward(dmlScript:StringBuilder, prefix:String) = {
    
    val out2:String = if(isNumber(outputShape._2)) "ignore1_"+id else outputShape._2
    val out3:String = if(isNumber(outputShape._3)) "ignore2_"+id else outputShape._3
    
    dmlScript.append(prefix).append(
      "[" + commaSep(outVar, out2, out3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayer.outVar, weight, bias, numChannels,  bottomLayer.outputShape._2, bottomLayer.outputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w) + ")\n")
  }
  
  override def outputShape = ( numKernels, outputHeight, outputWidth )
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace +
      "backward(" + commaSep(topLayer.dOut, outputShape._2, outputShape._3, bottomLayer.outVar, weight, bias,
                  numChannels, bottomLayer.outputShape._2, bottomLayer.outputShape._3, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)+ ")\n")
  override def weight = "W" + id
  override def bias = "b" + id
  override def dW = "dW" + id
  override def dB = "db" + id
  // -------------------------------------------------
  val outputHeight = try { ConvolutionUtils.getP(bottomLayer.outputShape._2.toLong, kernel_h.toLong, stride_h.toLong, pad_h.toLong).toString } catch { case _ : Throwable => "Hout" + id } 
  val outputWidth =  try { ConvolutionUtils.getQ(bottomLayer.outputShape._3.toLong, kernel_w.toLong, stride_w.toLong, pad_w.toLong).toString } catch { case _ : Throwable => "Wout" + id }
  def convParam = param.getConvolutionParam
  def numKernels = convParam.getNumOutput.toString
  def numChannels = bottomLayer.outputShape._1
  def kernel_h = if(convParam.hasKernelH) convParam.getKernelH.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def kernel_w = if(convParam.hasKernelW) convParam.getKernelW.toString 
                   else if(convParam.getKernelSizeCount > 0)  convParam.getKernelSize(0).toString 
                   else throw new LanguageException("Incorrect kernel parameters")
  def stride_h = if(convParam.hasStrideH) convParam.getStrideH.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else throw new LanguageException("Incorrect stride parameters")
  def stride_w = if(convParam.hasStrideW) convParam.getStrideW.toString 
                   else if(convParam.getStrideCount > 0)  convParam.getStride(0).toString 
                   else throw new LanguageException("Incorrect stride parameters")
  def pad_h =   if(convParam.hasPadH) convParam.getPadH.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else throw new LanguageException("Incorrect pad parameters")
  def pad_w =   if(convParam.hasPadW) convParam.getPadW.toString 
                   else if(convParam.getPadCount > 0)  convParam.getPad(0).toString 
                   else throw new LanguageException("Incorrect pad parameters")
}
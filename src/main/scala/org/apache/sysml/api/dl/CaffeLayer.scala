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

trait CaffeLayer {
  // -------------------------------------------------
  // Any layer that wants to reuse SystemML-NN has to override following methods that help in generating the DML for the given layer:
  def sourceFileName:String;
  def init(dmlScript:StringBuilder):Unit;
  def forward(dmlScript:StringBuilder):Unit;
  def backward(dmlScript:StringBuilder):Unit;
  def outputShape:(String, String, String) = bottomLayer.outputShape
  // -------------------------------------------------
  def bottomLayer:CaffeLayer = {
    val ret = net.getBottomLayers(param.getName).map(l => net.getCaffeLayer(l)).toList
    if(ret.size != 1) throw new LanguageException("Expected only 1 bottom layer")
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
  def param:LayerParameter
  def id:Int
  def net:CaffeNetwork
  def namespace:String = sourceFileName + "::"
  def outVar = "out" + id;
  def dOut = "dOut" + id;
  def weight = "W" + id
  def bias = "b" + id
  def dW = "dW" + id
  def dB = "db" + id
}

class Data(val param:LayerParameter, val id:Int, val net:CaffeNetwork, val numChannels:Int, val inputHeight:Int, val inputWidth:Int) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = null
  override def init(dmlScript:StringBuilder) = {
    dmlScript.append(" TODO: init" + param.getName)
  }
  override def forward(dmlScript:StringBuilder) = {
    dmlScript.append(" TODO: forward" + param.getName)
  }
  override def backward(dmlScript:StringBuilder) = {
    dmlScript.append(" TODO: backward" + param.getName)
  }
  override def outputShape = (numChannels.toString, inputHeight.toString, inputWidth.toString)
  // -------------------------------------------------
  def batchSize = param.getDataParam.getBatchSize
}

class SoftmaxWithLoss(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "softmax"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
      outVar + " = " + namespace + "forward(" + bottomLayer.outVar + ")\n")
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep("TODO", bottomLayer.outVar) + ")\n")
  // -------------------------------------------------
}

class ReLU(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "relu"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
      outVar + " = " + namespace + "forward(" + bottomLayer.outVar + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayer.dOut, bottomLayer.outVar) + ")\n")
  // -------------------------------------------------
}

class Dropout(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "dropout"
  override def init(dmlScript:StringBuilder) = { }
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
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
  override def init(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + "init(" + commaSep("TODO", numNeurons) + ")\n")
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
      outVar + " = " + namespace + "forward(" + commaSep(bottomLayer.outVar, weight, bias) + ")\n") 
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace + 
      "backward(" + commaSep(topLayer.dOut, bottomLayer.outVar, weight, bias) + ")\n")
  // -------------------------------------------------
  def numNeurons = param.getInnerProductParam.getNumOutput.toString
}

class MaxPooling(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "max_pool_builtin"
  override def init(dmlScript:StringBuilder) = {}
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(outVar, outputShape._2, outputShape._3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayer.outVar, numChannels,  bottomLayer.outputShape._2, bottomLayer.outputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w) + ")\n")
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      dOut + " = " + namespace + "backward(" + commaSep(topLayer.dOut, outputShape._2, outputShape._3, bottomLayer.outVar, 
                  numChannels, bottomLayer.outputShape._2, bottomLayer.outputShape._3, kernel_h, kernel_w, stride_h, stride_w)+ ")\n")
  override def outputShape = ( numChannels, "Hout" + id, "Wout" + id )
  // -------------------------------------------------
  def poolingParam = param.getPoolingParam
  def numChannels = bottomLayer.outputShape._2
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
}

class Convolution(val param:LayerParameter, val id:Int, val net:CaffeNetwork) extends CaffeLayer {
  // -------------------------------------------------
  override def sourceFileName = "conv_builtin";
  override def init(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(weight, bias) + "] = " + namespace + 
      "init(" + commaSep(numKernels, numChannels, kernel_h, kernel_w) + ")\n")
  override def forward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(outVar, outputShape._2, outputShape._3) + "] = " + namespace + 
      "forward(" + commaSep(bottomLayer.outVar, weight, bias, numChannels,  bottomLayer.outputShape._2, bottomLayer.outputShape._3, 
                            kernel_h, kernel_w, stride_h, stride_w) + ")\n")
  override def outputShape = ( numKernels, "Hout" + id, "Wout" + id )
  override def backward(dmlScript:StringBuilder) = dmlScript.append(
      "[" + commaSep(dOut, dW, dB) + "] = " + namespace +
      "backward(" + commaSep(topLayer.dOut, outputShape._2, outputShape._3, bottomLayer.outVar, weight, bias,
                  numChannels, bottomLayer.outputShape._2, bottomLayer.outputShape._3, kernel_h, kernel_w, stride_h, stride_w)+ ")\n")
  // -------------------------------------------------
  def convParam = param.getConvolutionParam
  def numKernels = convParam.getNumOutput.toString
  def numChannels = bottomLayer.outputShape._2
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
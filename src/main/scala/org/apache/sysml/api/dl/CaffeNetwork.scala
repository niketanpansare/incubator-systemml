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

import scala.collection.JavaConversions._
import caffe.Caffe.NetParameter
import caffe.Caffe.LayerParameter
import caffe.Caffe.Phase
import java.util.ArrayList
import java.util.HashSet
import scala.collection.mutable.Stack
import org.apache.sysml.parser.LanguageException;
import java.util.HashMap
import caffe.Caffe.PoolingParameter

// Wrapper on top of Caffe Network to simplify usage
// 
class CaffeNetwork(val currentPhase:Phase, val numChannels:Int, val inputHeight:Int, val inputWidth:Int) {
  def this(netFilePath:String, currentPhase:Phase, numChannels:Int, inputHeight:Int, inputWidth:Int) {
    this(currentPhase, numChannels, inputHeight, inputWidth)
    populateLayerNameList(Utils.readCaffeNet(netFilePath))
  }
  
  // Returns names of layers in sorted order
  def getLayers(): List[String] = layerNameList
  def getCaffeLayer(layerName:String):CaffeLayer = {
    val ret = layerNameMap.get(layerName)
    if(ret == null) throw new LanguageException("Layer with name " + layerName + " is not available for current phase: " + currentPhase.name() + ".")
    else ret
  }
  def getBottomLayers(layerName:String): HashSet[String] = layerNameBottomMap.get(layerName)
  def getTopLayers(layerName:String): HashSet[String] = layerNameTopMap.get(layerName)
  def getLayerID(layerName:String): Int = layerNameIDMap.get(layerName)
  
  private def getCaffeLayer(param:LayerParameter, id:Int) = {
    param.getType.toLowerCase() match {
      case "convolution" => new Convolution(param, id, this)
      case "pooling" => if(param.getPoolingParam.getPool == PoolingParameter.PoolMethod.MAX)  new MaxPooling(param, id, this)
                        else throw new LanguageException("Only maxpooling is supported:" + param.getPoolingParam.getPool.name)
      case "innerproduct" => new InnerProduct(param, id, this)
      case "relu" => new ReLU(param, id, this)
      case "softmaxwithloss" => new SoftmaxWithLoss(param, id, this)
      case "dropout" => new Dropout(param, id, this)
      case "data" => new Data(param, id, this, numChannels, inputHeight, inputWidth)
      case _ => throw new LanguageException("Layer of type " + param.getType + " is not supported")
    }
  }
  // ------------------------------------------------------------------
  private def getLayersFromCurrentPhase(net:NetParameter) = net.getLayerList.filter(l =>
	    if(l.getIncludeCount == 0) true else l.getIncludeList.filter(r => r.hasPhase() && r.getPhase != currentPhase).length == 0
	    // (l.getPhase == currentPhase)
	  )
	private var layerNameMap:HashMap[String, CaffeLayer] = new HashMap[String, CaffeLayer]
  private var layerNameBottomMap:HashMap[String, HashSet[String]] = new HashMap[String, HashSet[String]]
  private var layerNameTopMap:HashMap[String, HashSet[String]] = new HashMap[String, HashSet[String]]
  private var layerNameIDMap:HashMap[String, Int] = new HashMap[String, Int]
  private var layerNameList:List[String] = null
  private def populateLayerNameList(net:NetParameter):Unit = {
    // TODO: getTopologicalSortedLayers
    val tmp = getLayersFromCurrentPhase(net)
    var id = 1
    // First append all layerNameMap
    tmp.map(l => { 
      layerNameMap.put(l.getName, getCaffeLayer(l, id))
      layerNameIDMap.put(l.getName, id)
      id = id + 1
     })
    
    // Then append top/bottom layers that are available in layerNameMap
    tmp.map(l => {
      l.getBottomList.map(b => appendToHM(layerNameBottomMap, l.getName, b))
      l.getTopList.map(t => appendToHM(layerNameTopMap, l.getName, t))
    })
    layerNameList = tmp.map(_.getName).toList
  }
  private def appendToHM(hm:HashMap[String, HashSet[String]], key:String, value:String) = {
    if(!hm.containsKey(key)) hm.put(key, new HashSet[String]())
    // To include only top/bottom layers from current phase
    if(layerNameMap.containsKey(value))
      hm.get(key).add(value)
  }
  private def shouldVisit(layerName:String, visited:HashSet[String]):Boolean = {
    val iter = getBottomLayers(layerName).iterator()
    while(iter.hasNext()) {
      val bottomLayer = iter.next() 
      if(!bottomLayer.equals(layerName) && !visited.contains(bottomLayer)) {
        return false
      }
    }
    return true
	}
	private def getTopologicalSortedLayers(netLayersList:List[CaffeLayer]): List[CaffeLayer] = {
	  val visited:HashSet[String] = new HashSet[String]()
	  val ret:ArrayList[CaffeLayer] = new ArrayList[CaffeLayer]()
	  while(visited.size < netLayersList.size) {
	    var atleastOneVisited = false
	    for(l <- netLayersList) {
	      val isAlreadyVisited = visited.contains(l.param.getName)
	      if(!isAlreadyVisited && shouldVisit(l.param.getName, visited)) {
	        // System.out.print(">>" + l.getName)
	        visited.add(l.param.getName)
	        ret.add(l)
	        atleastOneVisited = true
	      }
	    }
	    if(!atleastOneVisited && visited.size < netLayersList.size) {
	      throw new LanguageException("Possible cycle")
	    }
	  }
	  ret.toList
	}
}
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

import java.net._
import java.util.concurrent.{Executors, ExecutorService}
import java.io.IOException
import java.io.{ObjectOutputStream, ObjectInputStream}
import java.util.{HashMap, HashSet, ArrayList}
import caffe.Caffe._
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import com.google.common.net.InetAddresses
import scala.collection.JavaConversions._
import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD

/**
 * This logic could be replaced later by either Spark streaming or Kafka streams or Storm or other streaming frameworks.
 * 
 * Advantage of this implementation:
 * - No additional dependency.
 * - Greater control over layer/weight placement.
 * 
 * Disadvantage of this implementation (as compared to generic streaming frameworks):
 * - Fault tolerance.
 * - Dynamic scalability.
 * - Restriction: only 1 executor per node.
 */
object StreamingGD {
  val LOG = LogFactory.getLog(classOf[StreamingGD].getName()) 
  // ------------------------------------------------------------------------------------
  // Global constants that user can modify.
  // Assumption for simplicity of implementation: one executor per node. This can be relaxed when we move to a generic streaming framework.
  var portNumber = 32897
  // At max, get messages from 100 layers at a time. This parameter might be important for wider networks and smaller cluster.
  var maxNumListenerPerNode = 100  
  // ------------------------------------------------------------------------------------
  
  // IP address of each layer is determined in the main
  var layerIPAddress:HashMap[String, String] = null
  var solver:SolverParameter = null
  var numChannels = -1
  var height = -1
  var width = -1
  // ------------------------------------------------------------------------------------
  // Returns msg with updated layer name and ip address.
  // Special cases:
  // 1. If last layer, this returns empty list
  // 2. If there are multiple output messages and only ip address change, then they share the inputActivation array
  // 3. If a layer needs multiple input messages, the process buffers the input messages until the last one arrives and returns empty for non-last message
  val buffer:HashMap[String, List[Message]] = new HashMap[String, List[Message]]
  def process(input:Message):List[Message] = {
    // TODO:
    return null
  }
  // ------------------------------------------------------------------------------------
  
  // train_solver.prototxt input_parquet_file output_weights_dir numChannels height width
  def main(args: Array[String]): Unit = {
    if(args.length != 6)
      throw new RuntimeException("Incorrect usage: train_solver.prototxt input_parquet_file output_weights_dir numChannels height width")
    val train_solver_file = args(0)
    val input_parquet_file = args(1)
    val output_weights_dir = args(2)
    numChannels = args(3).toInt
    height = args(4).toInt
    width = args(5).toInt
    
    val sc = new SparkContext()
    val spark = SparkSession.builder().getOrCreate()
    
    // Get IP Addresses
    val driverIPAddress = getLocalIpAddress()
    val executorIPAddresses = getIPAddressesAllNodes(sc).filter(!_.equals(driverIPAddress))
    
    // Get Layers
    solver = Utils.readCaffeSolver(train_solver_file)
    val layers = solver.getNetParam.getLayerList.filter(l => l.getPhase == caffe.Caffe.Phase.TRAIN).toList
    
    // Fill IP Address - Layer mapping
    // Assume the layers are topologically sorted and have correct layer/top/bottom naming
    layerIPAddress = new HashMap[String, String]
    if(!layers.get(0).getType.toLowerCase.equals("data")) {
      throw new RuntimeException("Expected the first layer to be of type data")
    }
    val batchSize = layers.get(0).getDataParam.getBatchSize
    layerIPAddress.put(layers.get(0).getName, driverIPAddress)
    
    // Simple scheme for partitioning the network
    simpleLayerMapping(layers, executorIPAddresses)
    LOG.info("The layer IP address mapping is as follows:" + layerIPAddress)
    
    // Make sure that layer mapping and solver is available on the executors
    distribute(sc, layerIPAddress, solver, numChannels, height, width, 2*(executorIPAddresses.length+1)*Runtime.getRuntime().availableProcessors())
    
    // Read input_parquet_file
    val inputRDD:RDD[(Array[Double], Int)] = readParquet(spark, input_parquet_file)
    
    val numEpochs = Math.ceil(solver.getMaxIter / batchSize).toInt
    for(epoch <- 0 until numEpochs) {
      extractBatch(inputRDD, batchSize)
    }
  }
  
  // ------------------------------------------------------------------------------------
  // Utility methods:
  private def extractBatch(inputRDD:RDD[(Array[Double], Int)], batchSize:Int):Unit = {
    val featuresBatch:Array[Array[Double]] = new Array[Array[Double]](batchSize)
    val labelsBatch:Array[Int] = new Array[Int](batchSize)
    var i = 0
    inputRDD.toLocalIterator.map(x => {
      featuresBatch(i) = x._1
      labelsBatch(i) = x._2
      if(i == batchSize - 1) {
        // Buffer full: send input batch
        sendInputBatch(featuresBatch, labelsBatch)
        i = 0
      }
      else {
        // Buffer input batch
        i = i + 1
      }
    })
    if(i > 0) {
      // Also send last minibatch
      val lastFeaturesBatch:Array[Array[Double]] = new Array[Array[Double]](i)
      val lastLabelsBatch:Array[Int] = new Array[Int](i)
      for(j <- 0 until i) {
        lastFeaturesBatch(j) = featuresBatch(j)
        lastLabelsBatch(j) = labelsBatch(j)
      }
      sendInputBatch(lastFeaturesBatch, lastLabelsBatch)
    }
  }
  
  // partitions the network such that every node has roughly equally number of layers
  private def simpleLayerMapping(layers:List[LayerParameter], executorIPAddresses:List[String]):Unit = {
    val numLayersPerExecutor = Math.ceil((layers.length-1).toDouble / executorIPAddresses.length).toInt
    var i = 1
    executorIPAddresses.map(ip => {
      for(j <- i until Math.min((i + numLayersPerExecutor), layers.length)) {
        layerIPAddress.put(layers.get(j).getName, ip)
      }
      i = i + numLayersPerExecutor
    })
  }
  
  // Sets StreamingGD.solver and StreamingGD.layerIPAddress on every executor
  private def distribute(sc:SparkContext, layerIPAddress:HashMap[String, String], solver:SolverParameter, 
      numChannels1:Int, height1:Int, width1:Int, numTasks:Int):Unit = {
    
    val layerIPAddressBroadcast = sc.broadcast(layerIPAddress)
    val solverBroadcast = sc.broadcast(solver)
    sc.parallelize(0 until numTasks).map(_ => {
      StreamingGD.synchronized {
        if(solver == null) {
          StreamingGD.solver = solverBroadcast.value
          StreamingGD.layerIPAddress = layerIPAddressBroadcast.value
          StreamingGD.numChannels = numChannels1
          StreamingGD.height = height1
          StreamingGD.width = width1
        }
      }
    }).count()
  }
  
  // Reads parquet file as RDD[Array[Double]]
  private def readParquet(spark:SparkSession, input_parquet_file:String):RDD[(Array[Double], Int)] = {
    spark.read.parquet(input_parquet_file).select("features", "labels").rdd.map(row => {
      if(row.get(0).isInstanceOf[org.apache.spark.mllib.linalg.SparseVector]) {
        (row.get(0).asInstanceOf[org.apache.spark.mllib.linalg.SparseVector].toArray, row.get(1).toString.toInt) 
      }
      else if(row.get(0).isInstanceOf[org.apache.spark.mllib.linalg.DenseVector]) {
        (row.get(0).asInstanceOf[org.apache.spark.mllib.linalg.DenseVector].toArray, row.get(1).toString.toInt)
      }
      else {
        throw new RuntimeException("Incorrect input parquet file:" + row.getClass)
      }
    }).persist(StorageLevel.MEMORY_AND_DISK)
  }
  
  private def getIPAddressesAllNodes(sc:SparkContext):List[String] = {
    val addresses = sc.getExecutorMemoryStatus.keys.map(ipPort => ipPort.split(":")(0))
    addresses.map(ip => if(InetAddresses.isInetAddress(ip)) ip else InetAddress.getByName(ip).getHostAddress).toList
  }
  @volatile var stop = false // Flag to stop the streaming server service
  private var _localIPAddress:String = null
  def getLocalIpAddress():String = {
    if(_localIPAddress == null) {
      _localIPAddress = InetAddress.getLocalHost.getHostAddress
    }
    return _localIPAddress
  }
  
  def sendMessage(msg:Message):Unit = {
    val newSocket = new Socket(StreamingGD.layerIPAddress.get(msg.layerName), StreamingGD.portNumber)
    val oos = new ObjectOutputStream(newSocket.getOutputStream())
    oos.writeObject(msg)
    newSocket.close
  }
  
  def sendInputBatch(features:Array[Array[Double]], labels: Array[Int]):Unit = {
    for(topLayerName <- solver.getNetParam.getLayer(0).getTopList) {
      sendMessage(new Message(features, labels, topLayerName, true))
    }
  }
}

class StreamingGD {
  
}

class Message(var inputActivation:Array[Array[Double]], var labels: Array[Int], var layerName:String, var isForward:Boolean) extends Serializable {
  def writeObject(out: ObjectOutputStream): Unit = {
    out.writeInt(inputActivation.length)
    if(inputActivation.length > 0) {
      out.writeInt(inputActivation(0).length)
      for(i <- 0 until inputActivation.length) {
        for(j <- 0 until inputActivation(i).length) {
          out.writeDouble(inputActivation(i)(j))
        }
      }
    }
    else {
      out.writeInt(0)
    }
    for(i <- 0 until labels.length) {
      out.writeInt(labels(i))
    }
    out.writeObject(layerName)
    out.writeBoolean(isForward)
  }
  def readObject(in:ObjectInputStream):Unit = {
    val batchSize = in.readInt()
    val numFeatures = in.readInt()
    inputActivation = new Array[Array[Double]](batchSize)
    for(i <- 0 until batchSize) {
      inputActivation(i) = new Array[Double](numFeatures)
      for(j <- 0 until numFeatures) {
        inputActivation(i)(j) = in.readDouble()
      }
    }
    labels = new Array[Int](batchSize)
    for(i <- 0 until batchSize) {
      labels(i) = in.readInt()
    }
    layerName = in.readObject().toString()
    isForward = in.readBoolean()
  }
}

class StreamingServerService extends Runnable {
  val serverSocket = new ServerSocket(StreamingGD.portNumber)
  val pool = Executors.newFixedThreadPool(StreamingGD.maxNumListenerPerNode)
  serverSocket.setSoTimeout(100)
  override def run():Unit = {
    try {
      while(StreamingGD.stop) {
       try {
         val socket = serverSocket.accept()
         pool.execute(new StreamingReceiver(socket))
       } catch {
         case e:SocketTimeoutException => {} // Do nothing. This allows us to gracefully stop the server socket.
       }
      }
    }
    catch {
      case e:Throwable  => throw new RuntimeException("Error occured in StreamingServerService:" + e.getMessage) 
    }
    finally {
      // Cleanup socket and threads
      pool.shutdownNow()
      try {
        serverSocket.close()
      } catch {
        case e:IOException => e.printStackTrace()
      }
      
    }
  }
}

class StreamingReceiver(val socket: Socket) extends Runnable {
  def isLocalLayer(msg:Message):Boolean = {
    val ipAddress = StreamingGD.layerIPAddress.get(msg.layerName)
    if(ipAddress == null) {
      throw new RuntimeException("No executor found for the layer: " + msg.layerName)
    }
    ipAddress.equals(StreamingGD.getLocalIpAddress)
  }
  override def run():Unit = {
    try {
      val ois = new ObjectInputStream(socket.getInputStream())
      var messages:List[Message] = List[Message](ois.readObject.asInstanceOf[Message])
      do {
        // Send remote messages to remote machines
        messages.filter(m => !isLocalLayer(m)).map(msg => StreamingGD.sendMessage(msg))
        // Process local messages
        messages = messages.filter(m => isLocalLayer(m)).flatMap(m => StreamingGD.process(m))
      } while(messages.length > 0);
      ois.close
    } finally {
      // Cleanup socket
      socket.close
    }
  }
}
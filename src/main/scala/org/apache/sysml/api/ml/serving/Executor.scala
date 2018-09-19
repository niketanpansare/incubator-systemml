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

package org.apache.sysml.api.ml.serving
import java.util.concurrent.PriorityBlockingQueue
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.runtime.instructions.gpu.context.GPUContext


case class Batch(size: Int, expectedTime: Long, priority: Double, modelName: String) extends Comparable[Batch] {
    override def compareTo(that: Batch): Int = {
        this.priority.compareTo(that.priority)
    }
}

class BatchQueue(execType: String, name: String) extends PriorityBlockingQueue[Batch] {
    private val expectedExecutionTime = new LongAdder()
    private var prevFirstRequest= Map[String, SchedulingRequest]()

    def getName : String = { name }

    def updatePrevRequest(name: String, request: SchedulingRequest) : Unit = {
        prevFirstRequest += (name -> request)
    }

    def getPrevRequest(name: String) : SchedulingRequest = { prevFirstRequest.getOrElse(name, null) }

    def enqueue(batch: Batch) : Unit = {
        synchronized {
            this.add(batch)
            expectedExecutionTime.add(batch.expectedTime)
        }
    }

    def dequeue() : Batch = {
        if (this.isEmpty)
            return Batch(-1, -1, -1, null)
        synchronized {
            val nextBatch = this.poll()
            expectedExecutionTime.add(-1*nextBatch.expectedTime)
            return nextBatch
        }
    }

    def getExpectedExecutionTime : Long = { expectedExecutionTime.longValue() }

    def getExecType : String = { execType }
}

class JmlcExecutor(scheduler: Scheduler, execType: String, gCtx: GPUContext, name: String) extends Runnable {
    @volatile protected var _shouldShutdown: Boolean = false

    var prevModel = ""

    def shutdown(): Unit = {
        _shouldShutdown = true
    }

    def getExecType: String = { execType }

    def getName: String = { name }

    def run(): Unit = {
        Thread.sleep(1000)
        while (!_shouldShutdown) {
            val requests = scheduler.schedule(this)
            if (requests.nonEmpty) {
                val responses = execute(requests)
                for ((req, resp) <- requests zip responses) {
                    req.response = resp
                    req.latch.countDown()
                }
            }
        }
    }

    def execute(requests: Array[SchedulingRequest]): Array[PredictionResponse] = {
        var responses = Array[PredictionResponse]()
        if (requests.nonEmpty) {
            try {
                val start = System.nanoTime()
                val batchedMatrixData = BatchingUtils.batchRequests(requests)
                val batchingTime = System.nanoTime() - start
                val req = requests(0)
                println("BEGIN PROCESS: " + req.model.name + " ON " + name)
                val modelAcquireStart = System.nanoTime()
                val script = scheduler.modelManager.acquire(req.model.name, execType, name)
                val modelAcquireTime = System.nanoTime() - modelAcquireStart
                script.setGpuContext(gCtx)
                script.setMatrix(req.model.inputVarName, batchedMatrixData, false)
                println("BEGIN EXEC: " + req.model.name + " ON " + name)
                val execStart = System.nanoTime()
                val res = script.executeScript().getMatrixBlock(req.model.outputVarName)
                val execTime = System.nanoTime() - execStart
                println("DONE EXEC: " + req.model.name + " ON " + name)
                responses = BatchingUtils.unbatchRequests(requests, res)
                val stop = System.nanoTime()
                val modelReleaseStart = System.nanoTime()
                scheduler.modelManager.release(req.model.name)
                scheduler.modelManager.releaseMemory(req.memUse)
                val modelReleaseTime = System.nanoTime() - modelReleaseStart
                scheduler.onCompleteCallback(req.model.name, stop - req.receivedTime, requests.length, execType)
                if (req.statistics != null)
                    setStatistics(requests, start, batchingTime, execTime, modelAcquireTime, modelReleaseTime)
                if (prevModel.nonEmpty)
                    prevModel = req.model.name
                println("DONE PROCESS: " + req.model.name + " ON " + name)
            } catch {
                case e: Exception => println("AN ERROR OCCURRED: " + e.getMessage + e.printStackTrace())
            }
        }
        responses
    }

    def setStatistics(requests: Array[SchedulingRequest],
                      processingStartTime: Long,
                      batchingTime: Long,
                      execTime: Long,
                      modelAcquireTime: Long,
                      modelReleaseTime: Long): Unit = {
        for (req <- requests) {
            req.statistics.batchingTime = batchingTime
            req.statistics.execType = getExecType
            req.statistics.batchSize = requests.length
            req.statistics.queueWaitTime = processingStartTime - req.receivedTime
            req.statistics.execTime = execTime
            req.statistics.modelAcquireTime = modelAcquireTime
            req.statistics.modelReleaseTime = modelReleaseTime
        }
    }
}
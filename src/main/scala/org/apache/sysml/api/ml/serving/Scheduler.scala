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

import scala.concurrent.Future
import scala.concurrent.duration._
import java.util.concurrent._
import java.util.concurrent.atomic.LongAdder

import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool

import scala.concurrent.ExecutionContext

case class SchedulingRequest(request: PredictionRequest,
                             model: Model,
                             latch: CountDownLatch,
                             receivedTime: Long,
                             var response: PredictionResponse = null,
                             statistics: RequestStatistics = null,
                             var memUse: Long = 0)

trait Scheduler {
    var executorService: ExecutorService = _
    protected var _statistics = true
    implicit val ec : ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(2000))
    var executorTypes = Array[String]()
    var modelManager = ReferenceCountedModelManager
    val requestsProcessed = new ConcurrentHashMap[String,LongAdder]()

    def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        val gCtxs = if (gpus != null) GPUContextPool.reserveAllGPUContexts() else null
        val numGpus = if (gCtxs != null) gCtxs.size() else 0
        executorService = Executors.newFixedThreadPool(numCores + numGpus)
        modelManager.setAvailableMemory((cpuMemoryBudgetInBytes*0.80).toLong)

        if (numCores > 0)
            executorTypes :+= "CPU"
        if (numGpus > 0)
            executorTypes :+= "GPU"

        println("STARTING SCHEDULER WITH: " + numCores + " CPU => " + numGpus + " GPUS")
        for (i <- 0 until numCores) {
            val exec = new JmlcExecutor(this, "CPU", null, "CPU" + i)
            executorQueues.put(exec, new BatchQueue("CPU", "CPU" + i))
            executorService.submit(exec)
        }
        for (i <- 0 until numGpus) {
            val exec = new JmlcExecutor(this, "GPU", gCtxs.get(i), "GPU" + i)
            executorQueues.put(exec, new BatchQueue("GPU", "GPU" + i))
            executorService.submit(exec)
        }
    }

    def shutdown(): Unit = {
        executorService.shutdown()
    }

    def schedule(executor: JmlcExecutor): Array[SchedulingRequest]

    /**
      * Registers a model with this scheduler. This should be called before enqueueing requests
      * @param model Model object to be registered
      */
    def addModel(model: Model): Unit = {
        modelQueues.putIfAbsent(model.name, new LinkedBlockingDeque[SchedulingRequest]())
        latencyObjectives.putIfAbsent(model.name, model.latencyObjective)
        modelManager.put(model)
    }

    /**
      * Sets a flag indicating if detailed statistics should be gathered which profile the time spent
      * in various stages of the execution pipeline
      * @param flag Boolean flag indicating whether statistics should be gathered
      */
    def setStatistics(flag: Boolean): Unit = { _statistics = flag }

    def timeout: Duration

    /**
      * Method which is used to update scheduler state of execution of a batch. If necessary
      * objects implementing the Scheduler trait should override this method and implement any logic needed
      * to post-process execution after a batch
      *
      * @param model String indicating the name of the model which was just executed
      * @param latency A measure of latency for this batch
      * @param batchSize The number of examples in the batch
      * @param execType The device type on which the batch was executed
      */
    def onCompleteCallback(model: String, latency: Double, batchSize: Int, execType: String) : Unit

    val requestQueue = new LinkedBlockingDeque[SchedulingRequest]()
    val globalSchedulingQueues = new ConcurrentHashMap[String, BatchQueue]()
    var modelQueues = new ConcurrentHashMap[String, BlockingQueue[SchedulingRequest]]()
    var executorQueues = new ConcurrentHashMap[JmlcExecutor, BatchQueue]()
    val dummyResponse = PredictionResponse(null, -1, null)
    val latencyObjectives = new ConcurrentHashMap[String, Duration]()
    var counter = 0

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction
      * @return
      */
    private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse]
}


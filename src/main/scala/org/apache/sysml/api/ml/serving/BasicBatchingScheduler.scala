package org.apache.sysml.api.ml.serving

import java.util.concurrent.{ConcurrentHashMap, CountDownLatch}

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.math.min

class BasicBatchingScheduler(override val timeout: Duration) extends BatchingScheduler {
    /**
      * Returns a list of requests to execute. If the list contains more than one element, they will be batched
      * by the executor. Returns an empty list when there are no models to be scheduled.
      * @param executor an Executor instance
      * @return a list of model requests to process
      */
    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        val execType = executor.getExecType
        dummyResponse.synchronized {
            val schedulableModels = getSchedulableModels(execType)
            if (schedulableModels.nonEmpty) {
                val (nextModel, nextBatchSize) = getNextModelAndBatchSize(schedulableModels, execType)
                for (_ <- 0 until nextBatchSize) {
                    val next = modelQueues.get(nextModel).poll()
                    assert(next != null, "Something is wrong")
                    ret :+= next
                }
            }
        }
        ret
    }

    /**
      * Helper method which gets the next model to schedule and the optimal batchsize
      * @param models A list of models to schedule
      * @return The model to schedule next
      */
    def getNextModelAndBatchSize(models : Iterable[String], execType: String) : (String, Int) = {
        val nextModel = models.map(
            m => ((getExpectedExecutionTime(m, getOptimalBatchSize(m, execType), execType)), m)
        ).minBy(x => x._1)._2

        val nextBatchSize = min(modelQueues.get(nextModel).size(),
            getOptimalBatchSize(nextModel, execType))
        (nextModel, nextBatchSize)
    }

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction
      * @return
      */
    override private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        statistics.queueSize = modelQueues.get(model.name).size
        modelQueues.get(model.name).add(schedulingRequest)
        counter += 1
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }

}

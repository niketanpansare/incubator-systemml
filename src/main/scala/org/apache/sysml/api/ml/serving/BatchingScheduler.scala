package org.apache.sysml.api.ml.serving
import java.util.concurrent.ConcurrentHashMap
import scala.math.{floor, max}

trait BatchingScheduler extends Scheduler {

    val modelBatchSizes = new ConcurrentHashMap[String, ConcurrentHashMap[String,Int]]()

    def getOptimalBatchSize(model : String, execType: String) : Int = {
        modelBatchSizes.putIfAbsent(execType, new ConcurrentHashMap[String,Int]())
        modelBatchSizes.get(execType).putIfAbsent(model, 1)
        modelBatchSizes.get(execType).get(model)
    }

    override def onCompleteCallback(model: String,
                                    latency: Double,
                                    batchSize: Int,
                                    execType: String): Unit = {
        val latencyObjective = latencyObjectives.get(model)
        val prevSize = modelBatchSizes.get(execType).get(model)
        if ((latency != latencyObjective.toNanos) && (batchSize == prevSize)) {
            modelBatchSizes.synchronized({
                modelBatchSizes.get(execType).put(model,
                    if (latency < latencyObjective.toNanos) prevSize+2 else max(floor(prevSize*0.90).toInt, 1))
            })
        }

    }

    def getExpectedExecutionTime(model: String, batchSize: Int, execType: String) : Long = {
        latencyObjectives.get(model).toNanos
    }

    /**
      * Gets a list of models that are eligible to be run. A model is eligible to be run if it
      * has a greater number of requests enqueued than its optimal batch size.
      * @return A list of models which may be scheduled
      */
    def getSchedulableModels(execType: String) : Set[String] = {
        var batchableModels = Set[String]()
        var shortFuse = Set[String]()
        val keyIterator = modelQueues.keys()
        while (keyIterator.hasMoreElements) {
            val name = keyIterator.nextElement()
            if (modelQueues.get(name).size() > 0) {
                val nextRequest = modelQueues.get(name).peek()
                assert(nextRequest != null, "Something is wrong. Next request should not be null")

                if (checkShortFuse(nextRequest)) {
                    shortFuse += name
                }

                if (modelQueues.get(name).size() >= getOptimalBatchSize(name, execType)) {
                    batchableModels += name
                }
            }
        }

//        var shortFuseArray = Set[String]()
//        if (shortFuse.nonEmpty) {
//            shortFuseArray += shortFuse.minBy(x => x._2)._1
//        }

        if (shortFuse.nonEmpty) shortFuse else batchableModels
    }

    /**
      * Returns a boolean value indicating whether or not a request has exceeded its latency objective by more
      * than 20%
      */
    def checkShortFuse(request: SchedulingRequest) : Boolean = {
        val elapsed = System.nanoTime() - request.receivedTime
        val shortFuse = elapsed >= (1.2*latencyObjectives.get(request.model.name).toNanos).toLong
        shortFuse
    }
}

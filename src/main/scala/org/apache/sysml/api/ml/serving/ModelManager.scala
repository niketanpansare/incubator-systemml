package org.apache.sysml.api.ml.serving

import java.util.concurrent.atomic.LongAdder
import org.apache.sysml.api.jmlc.PreparedScript
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.utils.PersistentLRUCache

trait ModelManager {

    val availableMemory = new LongAdder

    var totalMemory = 0L

    var cleanupEnabled = true

    var memCheckEnabled = true

    var models: Map[String, Model] = Map()

    def setAvailableMemory(memBytes: Long) : Unit = {
        println("SETTING TOTAL MEMORY AVAILABLE TO: " + memBytes)
        totalMemory = memBytes
        availableMemory.reset()
        availableMemory.add(memBytes)
    }

    def getAvailableMemory : Long = { availableMemory.longValue() }

    def acquireMemory(bytes: Long) : Long = {
        // if memory checking is not enabled just always say they get the memory
        if (!memCheckEnabled || bytes == 0)
            return bytes

        // otherwise check to see if there is enough memory to meet the request
        if (bytes <= availableMemory.longValue()) {
            this.synchronized {
                if (bytes <= availableMemory.longValue()) {
                    println("GRANTED: " + bytes + "/" + availableMemory)
                    availableMemory.add(-1 * bytes)
                    return bytes
                }
            }
        }
        // not enough memory available :(
        -1
    }

    def releaseMemory(bytes: Long) : Unit = {
        if (bytes > 0) {
            println("RELEASING: " + bytes)
            availableMemory.add(bytes)
            println("MEMORY IS NOW: " + availableMemory.longValue())
        }
    }

    def disableCleanup() : Unit = { cleanupEnabled = false }

    def disableMemcheck() : Unit = { memCheckEnabled = false }

    def put(model: Model): Unit

    def get(name: String): Model

    def putWeight(name: String, weight: MatrixBlock) : Unit

    def acquire(name: String, execType: String, execName: String) : PreparedScript

    def release(name: String) : Unit
}

object ReferenceCountedModelManager extends ModelManager {
    var modelRefCounts: Map[String,LongAdder] = Map()
    var weightCache : PersistentLRUCache = _
//    val weightCache = new ConcurrentHashMap[String,MatrixBlock]()

    override def setAvailableMemory(maxBytes: Long) : Unit = {
        super.setAvailableMemory(maxBytes)
        weightCache = new PersistentLRUCache((0.80*maxBytes).toLong)
        weightCache.enableReadOnlyMode(true)
    }

    def tryAllocMem(name: String, batchSize: Int) : Long = {
        val extraMem = (0.5*models(name).weightMem).toLong
        val weightMem = if (modelRefCounts(name).longValue() > 0) 0L else models(name).weightMem
        val memReceived = acquireMemory(extraMem + weightMem)
        if (memReceived < 0) memReceived else extraMem
    }

    def isCached(name: String) : Boolean = { modelRefCounts(name).longValue() > 0 }

    def acquire(name: String, execType: String, execName: String) : PreparedScript = {
         println("ACQUIRING MODEL: " + name + " => " + modelRefCounts(name).longValue())
        // if the model has non-zero refcount then all weights are
        // guaranteed to be already pinned, so we can return immediately
        val ps = models(name).script(execType)
        if (modelRefCounts(name).longValue() > 0 && ps.hasPinnedVars) {
            modelRefCounts(name).increment()
            return ps.clone(false)
        }

        // otherwise we need to re-pin the weights, possibly reading them from disk
        println("READING: " + name + " FROM CACHE. MEM AVAILABLE => " + weightCache.getNumBytes)
        val model = models(name)
        model.synchronized {
            if (modelRefCounts(name).longValue() == 0)
                model.weightFiles.foreach(x => ps.setMatrix(x._1, weightCache.getAsMatrixBlock(x._2), true))
            modelRefCounts(name).increment()
        }
        println("DONE ACQUIRING MODEL: " + name)
        ps.clone(false)
    }

    override def disableCleanup(): Unit = {
        super.disableCleanup()
        println("CLEANUP IS DISABLED")
    }

    def release(name: String) : Unit = {
        modelRefCounts(name).decrement()

        println("RELEASE MODEL: " + name + " => " + modelRefCounts(name).longValue())
        if (modelRefCounts(name).longValue() == 0 && cleanupEnabled) {
            models(name).synchronized {
                if (modelRefCounts(name).longValue() == 0) {
                    println("ACTUALLY RELEASING THE MODEL")
                    models(name).script.foreach { x => x._2.clearInVarReuse() }
                    models(name).script.foreach { x => x._2.clearParameters() }
                    println("CALLING RELEASE MEMORY")
                    releaseMemory(models(name).weightMem)
                }
            }
        }
    }

    def put(model: Model) : Unit = {
        models += (model.name -> model)
        modelRefCounts += (model.name -> new LongAdder())
    }

    def putWeight(name: String, weight: MatrixBlock) : Unit = {
        weightCache.put(name, weight)
    }

    def get(name: String) : Model = { models(name) }

}

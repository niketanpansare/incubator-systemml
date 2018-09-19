package org.apache.sysml.api.ml.serving
import org.apache.sysml.runtime.matrix.data.MatrixBlock

object BatchingUtils {
        def batchRequests(requests: Array[SchedulingRequest]) : MatrixBlock = {
            if (requests.length == 1) {
                return requests(0).request.data
            }
            val ncol = requests(0).request.data.getNumColumns
            val res = new MatrixBlock(requests.length, ncol, -1).allocateDenseBlock()
            val doubles = res.getDenseBlockValues
            var start = 0
            for (req <- requests) {
                System.arraycopy(req.request.data.getDenseBlockValues, 0, doubles, start, ncol)
                start += ncol
            }
            res.setNonZeros(-1)
            res
        }

        def unbatchRequests(requests: Array[SchedulingRequest],
                            batchedResults: MatrixBlock) : Array[PredictionResponse] = {
            var responses = Array[PredictionResponse]()
            val start = 0
            for (req <- requests) {
                val unbatchStart = System.nanoTime()
                val resp = PredictionResponse(batchedResults.slice(
                    start, (start + req.request.requestSize)-1), 
                    batchedResults.getNumRows, req.statistics)
                val unbatchingTime = System.nanoTime() - unbatchStart
                if (req.statistics != null)
                    req.statistics.unbatchingTime = unbatchingTime

                responses :+= resp
            }

            responses
        }
}

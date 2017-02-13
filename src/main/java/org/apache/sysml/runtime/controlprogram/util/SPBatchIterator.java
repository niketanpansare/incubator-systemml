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

package org.apache.sysml.runtime.controlprogram.util;

import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.functions.ExtractBlockForBinaryReblock;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

public class SPBatchIterator extends CPBatchIterator {
	MatrixCharacteristics mcIn;
	JavaPairRDD<MatrixIndexes,MatrixBlock> in1;
	SparkExecutionContext sec;
	
	public SPBatchIterator(ExecutionContext ec, String[] iterablePredicateVars) throws DMLRuntimeException {
		super(ec, iterablePredicateVars);
		sec = ((SparkExecutionContext)ec);
		mcIn = new MatrixCharacteristics(X.getNumRows(), X.getNumColumns(), (int)batchSize, (int)X.getNumColumns());
		in1 = getReblockedMatrix( iterablePredicateVars[1], mcIn ).persist(StorageLevel.MEMORY_AND_DISK());
		if(isPrefetchEnabled)
			startPrefetch();
	}
	
	private JavaPairRDD<MatrixIndexes,MatrixBlock> getReblockedMatrix(String varName, MatrixCharacteristics mcOut) throws DMLRuntimeException {
		JavaPairRDD<MatrixIndexes,MatrixBlock> temp = sec.getBinaryBlockRDDHandleForVariable( varName );
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(varName);
		temp = RDDAggregateUtils.mergeByKey(temp.flatMapToPair(new ExtractBlockForBinaryReblock(mcIn, mcOut)));
		return temp;
	}
	
	MatrixObject nextBatch = null;
	Thread prefetchThread = null;
	
	public void startPrefetch() {
		if(hasNext()) {
			prefetchThread = new Thread(new PrefetchingThread(this));
			prefetchThread.start();
		}
		else
			prefetchThread = null;
	}
	
	class PrefetchingThread implements Runnable {
		SPBatchIterator iter;
		public PrefetchingThread(SPBatchIterator iter) {
			this.iter = iter;
		}
		@Override
		public void run() {
			long beg = (iter.currentBatchIndex * iter.batchSize) % iter.N + 1;
			iter.currentBatchIndex++;
			long end = Math.min(iter.N, beg + iter.batchSize - 1);
			iter.nextBatch = getNextBatch(beg, end);
		}
	}
	
	@Override
	public MatrixObject next() {
		if(isPrefetchEnabled) {
			if(prefetchThread != null) {
				// Prefetching in-progress
				try {
					prefetchThread.join();
				} catch (InterruptedException e) {
					throw new RuntimeException("Prefetching thread is interrupter.", e);
				}
				if(nextBatch != null) {
					MatrixObject currentBatch = nextBatch;
					nextBatch = null;
					startPrefetch();
					return currentBatch;
				}
				else {
					throw new RuntimeException("No block returned by prefetching thread.");
				}
			}
			else {
				throw new RuntimeException("No prefetching thread set.");
			}
		}
		else {
			long beg = (currentBatchIndex * batchSize) % N + 1;
			currentBatchIndex++;
			long end = Math.min(N, beg + batchSize - 1);
			return getNextBatch(beg, end);
		}
	}
	
	
	private MatrixObject getNextBatch(long beg, long end) {
		IndexRange ixrange = new IndexRange(beg-1, end-1, 0, X.getNumColumns()-1);
		MatrixObject ret = null;
		
		try {
			// Perform the operation (no prefetching for CP as X.acquireRead() might or might not fit on memory): 
			// # Get next batch
		    // beg = ((i-1) * batch_size) %% N + 1
		    // end = min(N, beg + batch_size - 1)
		    // X_batch = X[beg:end,]
			
			//single block output via lookup (on partitioned inputs, this allows for single partition
			//access to avoid a full scan of the input; note that this is especially important for 
			//out-of-core datasets as entire partitions are read, not just keys as in the in-memory setting.
			long rix = UtilFunctions.computeBlockIndex(ixrange.rowStart, mcIn.getRowsPerBlock());
			long cix = UtilFunctions.computeBlockIndex(ixrange.colStart, mcIn.getColsPerBlock());
			List<MatrixBlock> list = in1.lookup(new MatrixIndexes(rix, cix));
			if( list.size() != 1 )
				throw new DMLRuntimeException("Block lookup returned "+list.size()+" blocks (expected 1).");
			
			MatrixBlock tmp = list.get(0);
//			MatrixBlock mbout = (tmp.getNumRows()==mcOut.getRows() && tmp.getNumColumns()==mcOut.getCols()) ? 
//					tmp : tmp.sliceOperations( //reference full block or slice out sub-block
//					UtilFunctions.computeCellInBlock(ixrange.rowStart, mcIn.getRowsPerBlock()), 
//					UtilFunctions.computeCellInBlock(ixrange.rowEnd, mcIn.getRowsPerBlock()), 
//					UtilFunctions.computeCellInBlock(ixrange.colStart, mcIn.getColsPerBlock()), 
//					UtilFunctions.computeCellInBlock(ixrange.colEnd, mcIn.getColsPerBlock()), new MatrixBlock());
			
			
			// Return X_batch as MatrixObject
			MatrixCharacteristics mc = new MatrixCharacteristics(end - beg + 1, 
					X.getNumColumns(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
			ret = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
					new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
			ret.acquireModify(tmp);
			ret.release();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while fetching a batch", e);
		}
		return ret;
	}
	
}

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

import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.utils.Statistics;

public class CPBatchIterator implements Iterator<Data>, Iterable<Data> {
	
	protected MatrixObject X = null;
	protected long batchSize;
	protected long numBatches;
	protected long currentBatchIndex = 0;
	protected long N;
	protected boolean isPrefetchEnabled = false;
	protected String XName;
	
	private static final Log LOG = LogFactory.getLog(CPBatchIterator.class.getName());
	
	public static Iterable<Data> getBatchIterator(ExecutionContext ec, String[] iterablePredicateVars) throws DMLRuntimeException {
		MatrixObject X = (MatrixObject) ec.getVariable( iterablePredicateVars[1] );
		long maxNNZ = X.getNumRows()*X.getNumColumns();
		long batchSize = Long.parseLong(iterablePredicateVars[3]); 
		boolean canFitInCP = (maxNNZ+batchSize)*8 < InfrastructureAnalyzer.getLocalMaxMemory();
		
		if(ec instanceof SparkExecutionContext) {
			if(canFitInCP && !ParForProgramBlock.isInParForLoop())
				return (Iterable<Data>) (new CPBatchIterator(ec, iterablePredicateVars));
			else
				return (Iterable<Data>) (new SPBatchIterator(ec, iterablePredicateVars));
		}
		else {
			return (Iterable<Data>) (new CPBatchIterator(ec, iterablePredicateVars));
		}
	}
	
	
	public CPBatchIterator(ExecutionContext ec, String[] iterablePredicateVars) {
		X = (MatrixObject) ec.getVariable( iterablePredicateVars[1] );
		// assumption: known at runtime
		N = X.getNumRows();
		try {
			batchSize = Long.parseLong(iterablePredicateVars[3]); 
		} catch(NumberFormatException e) {  
			batchSize = ((ScalarObject) ec.getVariable( iterablePredicateVars[3] )).getLongValue();
		}
		numBatches = (long) Math.ceil(  ((double)N) / batchSize);
		double prefetchMemoryBudgetInMB = ConfigurationManager.getDMLConfig().getDoubleValue(DMLConfig.PREFETCH_MEM_BUDGET);
		double requiredBudgetInMB = (batchSize*X.getNumColumns()*8)/1000000;
		if(ParForProgramBlock.isInParForLoop()) {
			LOG.warn("Prefetching is disabled if executed inside a parfor program block");
		}
		else if(prefetchMemoryBudgetInMB >= requiredBudgetInMB) {
			isPrefetchEnabled = true;
			LOG.info("Prefetching is enabled");
			System.out.println("Prefetching is enabled");
		}
		else if(prefetchMemoryBudgetInMB != 0) {
			LOG.warn("Prefetching is disabled as prefetch memory budget (" + prefetchMemoryBudgetInMB + " mb) is smaller than " + requiredBudgetInMB);
		}
	}

	@Override
	public Iterator<Data> iterator() {
		return this;
	}

	@Override
	public boolean hasNext() {
		return currentBatchIndex < numBatches;
	}

	@Override
	public MatrixObject next() {
		long startTime = DMLScript.STATISTICS ? System.nanoTime() : -1;
		long beg = (currentBatchIndex * batchSize) % N + 1;
		currentBatchIndex++;
		long end = Math.min(N, beg + batchSize - 1);
		IndexRange ixrange = new IndexRange(beg-1, end-1, 0, X.getNumColumns()-1);
		MatrixObject ret = null;
		try {
			// Perform the operation (no prefetching for CP as X.acquireRead() might or might not fit on memory): 
			// # Get next batch
		    // beg = ((i-1) * batch_size) %% N + 1
		    // end = min(N, beg + batch_size - 1)
		    // X_batch = X[beg:end,]
			MatrixBlock matBlock = X.acquireRead();
			MatrixBlock resultBlock = matBlock.sliceOperations(ixrange, new MatrixBlock());
			X.release();
			// Return X_batch as MatrixObject
			MatrixCharacteristics mc = new MatrixCharacteristics(end - beg + 1, 
					X.getNumColumns(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
			ret = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), 
					new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
			ret.acquireModify(resultBlock);
			ret.release();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while fetching a batch", e);
		}
		
		if(DMLScript.STATISTICS) {
			long endTime = System.nanoTime();
			Statistics.batchFetchingTimeInIndexing = endTime - startTime;
			Statistics.batchFetchingTimeInNext = Statistics.batchFetchingTimeInIndexing; 
		}
		return ret;
	}
}

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

package org.apache.sysml.runtime.matrix.data;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.CPPUtil;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.utils.Statistics;

public class LibMatrixNative {
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret)  throws DMLRuntimeException {
		//pre-processing: output allocation
		double [] retBlk = ret.getDenseBlock();
		if(retBlk == null || m1.getNumRows()*m2.getNumColumns() != retBlk.length)
			ret.allocateDenseBlock();
		retBlk = ret.getDenseBlock();
		
		if(m1.getDenseBlock() == null || m2.getDenseBlock() == null || retBlk == null)
			throw new DMLRuntimeException("Expected the input and outputs to be allocated in dense format");
		
		Statistics.numNativeCalls.addAndGet(1);
		CPPUtil.matrixMultDenseDense(m1.getDenseBlock(), m2.getDenseBlock(), retBlk, m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns());
		
		//post-processing: nnz/representation
		ret.recomputeNonZeros();
	
		ret.examSparsity();
	}
	
	/**
	 * 
	 * @param ec
	 * @param blk
	 * @return false if the program block cannot be executed natively
	 * @throws DMLRuntimeException
	 */
	public static boolean execute(ExecutionContext ec, ProgramBlock blk) throws DMLRuntimeException {
		if(DMLScript.ENABLE_NATIVE_LOOP) {
			// TODO: Cache CPPUtil to avoid cost of encoding if this block is not recompiled
			return new CPPUtil(ec).execute(blk);
		}
		return false;
	}
}

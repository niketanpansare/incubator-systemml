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
package org.apache.sysml.runtime.controlprogram.context;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class GPUContext {

	protected ArrayList<GPUPointer> allocatedPointers = new ArrayList<GPUPointer>(); 
	protected static GPUContext currContext;
	protected GPUContext() { }
	
	
	public static void createGPUContext() {
		if(currContext == null)
			currContext = new JCudaContext();
	}
	public static GPUContext getCurrentContext() {
		return currContext;
	}
	public abstract void destroy();
	public void prepareOutput(MatrixBlock mat) throws DMLRuntimeException {
		prepare(mat, false, true);
	}
	// Copying from device <- host occurs here
	public abstract void copyDeviceToHost(MatrixBlock mat) throws DMLRuntimeException;
	/**
	 * This method removes GPUPointers and related information from GPUContext.
	 * This method is called by rmvar, (cleanup for updateInPlace MatrixObject), CP setMatrixOutput and while GPUContext's internal eviction.  
	 * 
	 * @param mat
	 * @throws DMLRuntimeException
	 */
	public abstract void remove(MatrixBlock mat) throws DMLRuntimeException;
	
	
	// Copying from device -> host occurs here
	/**
	 * This function should prepares GPUPointer for processing of GPUInstruction. 
	 * If the MatrixBlock is to be used as input, then it should copy the data from host to device. 
	 * Locking the MatrixBlock ensures that it won't be evicted during the execution.
	 * 
	 */
	abstract void prepare(MatrixBlock mat, boolean isInput, boolean lock) throws DMLRuntimeException;
	abstract void unlock(MatrixBlock mat, boolean isGPUCopyModified);
	
	
	
	
	
	// Operations to be implemented by every GPUContext
	public abstract void conv2d(MatrixBlock image, MatrixBlock filter, MatrixBlock output, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q) throws DMLRuntimeException;
	public abstract void conv2d_backward_filter(MatrixBlock image, MatrixBlock dout, MatrixBlock output, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q) throws DMLRuntimeException;
	public abstract void matmult(MatrixBlock left, MatrixBlock right, MatrixBlock output, boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException;
	
}

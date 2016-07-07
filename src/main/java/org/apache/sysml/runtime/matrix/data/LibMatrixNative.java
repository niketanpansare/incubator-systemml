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
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN.ConvolutionParameters;
import org.apache.sysml.utils.Statistics;

public class LibMatrixNative {
	
	static {
		// Load native library at runtime
		// SystemML.dll (Windows) or libSystemML.so (Unix)
		if(DMLScript.USE_NATIVE)
			System.loadLibrary("SystemML");
	}

	private static native void matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen);
	private static native void conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q);
	private static native void conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q);
	
	
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k)  throws DMLRuntimeException {
		if(m1.isInSparseFormat() || m2.isInSparseFormat()) {
			LibMatrixMult.matrixMult(m1, m2, ret, k);
		}
		else {
			//pre-processing: output allocation
			ret.sparse = (m1.isUltraSparse() || m2.isUltraSparse());
			if( !ret.sparse )
				ret.allocateDenseBlock();
			
			if(m1.denseBlock == null || m2.denseBlock == null || ret.denseBlock == null)
				throw new DMLRuntimeException("Expected the input and outputs to be allocated in dense format");
			
			Statistics.numNativeCalls.addAndGet(1);
			matrixMultDenseDense(m1.denseBlock, m2.denseBlock, ret.denseBlock, m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns());
			
			//post-processing: nnz/representation
			// if( !ret.sparse )
			ret.recomputeNonZeros();
			ret.examSparsity();
		}
	}
	
	public static void conv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				filter.getNumRows() != params.K || filter.getNumColumns() != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input to conv2d");
		}
		
		if(DMLScript.USE_NATIVE && !input.isInSparseFormat() && !filter.isInSparseFormat()) {
			Statistics.numNativeCalls.addAndGet(1);
			LibMatrixNative.conv2dDense(input.denseBlock, filter.denseBlock, outputBlock.denseBlock, 
					params.N, params.C, params.H, params.W, params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, 
					params.pad_w, params.P, params.Q);
		}
		else {
			throw new DMLRuntimeException("Sparse native conv2d is not supported");
		}
	}
	
	public static void conv2d_backward_filter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, ConvolutionParameters params) throws DMLRuntimeException {
		if(input.getNumRows() != params.N || input.getNumColumns() != params.C*params.H*params.W || 
				dout.getNumRows() != params.N || dout.getNumColumns() != params.K*params.P*params.Q) {
			throw new DMLRuntimeException("Incorrect input to conv2d_backward_filter");
		}
		
		if(DMLScript.USE_NATIVE && !input.isInSparseFormat() && !dout.isInSparseFormat()) {
			Statistics.numNativeCalls.addAndGet(1);
			LibMatrixNative.conv2dBackwardFilterDense(input.denseBlock, dout.denseBlock, outputBlock.denseBlock, 
					params.N, params.C, params.H, params.W, params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, 
					params.pad_w, params.P, params.Q);
		}
		else {
			throw new DMLRuntimeException("Sparse native conv2d_backward_filter is not supported");
		}
	}
}

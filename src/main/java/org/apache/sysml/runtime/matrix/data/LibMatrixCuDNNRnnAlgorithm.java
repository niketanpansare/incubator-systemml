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

import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensorNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensorNdDescriptorEx;
import static jcuda.jcudnn.JCudnn.cudnnDestroyDropoutDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyRNNDescriptor;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcudnn.JCudnn.cudnnCreateRNNDescriptor;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;

import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnRNNDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

public class LibMatrixCuDNNRnnAlgorithm implements java.lang.AutoCloseable {
	GPUContext gCtx;
	String instName;
	cudnnDropoutDescriptor dropoutDesc;
	cudnnRNNDescriptor rnnDesc;
	cudnnTensorDescriptor[] xDesc, yDesc;
	cudnnTensorDescriptor hxDesc, cxDesc, hyDesc, cyDesc;
	cudnnFilterDescriptor wDesc;
	long sizeInBytes; Pointer workSpace;
	long reserveSpaceSizeInBytes; Pointer reserveSpace;
	public LibMatrixCuDNNRnnAlgorithm(GPUContext gCtx, String instName, String rnnMode, int N, int T, int M, int D, boolean isTraining) throws DMLRuntimeException {
		this.gCtx = gCtx;
		this.instName = instName;
		initializeDropout();
		initializeRnnDescriptor(rnnMode, N, T, M, D);
		
		long [] sizeInBytesArray = new long[1]; long [] reserveSpaceSizeInBytesArray = new long[1];
		workSpace = new Pointer(); reserveSpace = new Pointer();
		JCudnn.cudnnGetRNNWorkspaceSize(gCtx.getCudnnHandle(), rnnDesc, T, xDesc, sizeInBytesArray);
		sizeInBytes = sizeInBytesArray[0];
		if(sizeInBytes != 0)
			workSpace = gCtx.allocate(sizeInBytes);
		if(isTraining) {
			JCudnn.cudnnGetRNNTrainingReserveSize(gCtx.getCudnnHandle(), rnnDesc, T, xDesc, reserveSpaceSizeInBytesArray);
			reserveSpaceSizeInBytes = reserveSpaceSizeInBytesArray[0];
			if (reserveSpaceSizeInBytes != 0)
				reserveSpace = gCtx.allocate(reserveSpaceSizeInBytes);
		}
	}
	
	private void initializeRnnDescriptor(String rnnMode, int N, int T, int M, int D) throws DMLRuntimeException {
		int inputMode = jcuda.jcudnn.cudnnRNNInputMode.CUDNN_LINEAR_INPUT; // alternative: CUDNN_SKIP_INPUT 
		int rnnModeVal = -1;
		if(rnnMode.equalsIgnoreCase("rnn_relu")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_RELU;
		}
		else if(rnnMode.equalsIgnoreCase("rnn_tanh")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_RNN_TANH;
		}
		else if(rnnMode.equalsIgnoreCase("lstm")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_LSTM;
		}
		else if(rnnMode.equalsIgnoreCase("gru")) {
			rnnModeVal = jcuda.jcudnn.cudnnRNNMode.CUDNN_GRU;
		}
		else {
			throw new DMLRuntimeException("Unsupported rnn mode:" + rnnMode);
		}
		rnnDesc = new cudnnRNNDescriptor();
		cudnnCreateRNNDescriptor(rnnDesc);
		int rnnAlgo = jcuda.jcudnn.cudnnRNNAlgo.CUDNN_RNN_ALGO_STANDARD; // TODO:
		JCudnn.cudnnSetRNNDescriptor(gCtx.getCudnnHandle(), rnnDesc, M, 1, dropoutDesc, inputMode, jcuda.jcudnn.cudnnDirectionMode.CUDNN_UNIDIRECTIONAL, 
				rnnModeVal, rnnAlgo, LibMatrixCUDA.CUDNN_DATA_TYPE);
		xDesc = new cudnnTensorDescriptor[] {allocateTensorDescriptor(N, D, 1)};
		hxDesc = allocateTensorDescriptor(1, N, M); 
		cxDesc = allocateTensorDescriptor(1, N, M);
		yDesc = new cudnnTensorDescriptor[T];
		for(int t = 0; t < T; t++) {
			yDesc[t] = allocateTensorDescriptorWithColumnStride(N, D);
		}
		hyDesc = allocateTensorDescriptor(1, N, M);
		cyDesc = allocateTensorDescriptor(1, N, M);
		long [] weightSizeInBytesArray = {-1};
		JCudnn.cudnnGetRNNParamsSize(gCtx.getCudnnHandle(), rnnDesc, xDesc[0], weightSizeInBytesArray, LibMatrixCUDA.CUDNN_DATA_TYPE);
		// check if (D+M+2)*4M == weightsSize / sizeof(dataType) where weightsSize is given by 'cudnnGetRNNParamsSize'.
		int expectedNumWeights = LibMatrixCUDA.toInt(weightSizeInBytesArray[0]/LibMatrixCUDA.sizeOfDataType);
		if(rnnMode.equalsIgnoreCase("lstm") && (D+M+2)*4*M != expectedNumWeights) {
			throw new DMLRuntimeException("Incorrect number of RNN parameters " +  (D+M+2)*4*M + " != " +  expectedNumWeights + ", where numFeatures=" + D + ", hiddenSize=" + M);
		}
		wDesc = allocateFilterDescriptor(expectedNumWeights, 1, 1);
		
		
	}
	
	private void initializeDropout() throws DMLRuntimeException {
		float dropout = 1.0f;
		long randSeed = -1; //  new Random().nextLong();
		// Dropout descriptor
		dropoutDesc = new cudnnDropoutDescriptor();
		JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
		long [] dropOutSizeInBytes = {-1};
		JCudnn.cudnnDropoutGetStatesSize(gCtx.getCudnnHandle(), dropOutSizeInBytes);
		Pointer dropOutStateSpace = new Pointer();
		if (dropOutSizeInBytes[0] != 0)
			dropOutStateSpace = gCtx.allocate(dropOutSizeInBytes[0]);
		JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, gCtx.getCudnnHandle(), dropout, dropOutStateSpace, dropOutSizeInBytes[0], randSeed);
	}
	
	private static cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H) throws DMLRuntimeException {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		cudnnSetTensorNdDescriptorEx(tensorDescriptor, CUDNN_TENSOR_NCHW, LibMatrixCUDA.CUDNN_DATA_TYPE, 3, new int[] {N, C, H});
		return tensorDescriptor;
	}
	
	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		JCudnn.cudnnSetFilterNdDescriptor(filterDesc, LibMatrixCUDA.CUDNN_DATA_TYPE, CUDNN_TENSOR_NCHW, 3, new int[] {K, C, R});
		return filterDesc;
	}
	
	private static cudnnTensorDescriptor allocateTensorDescriptorWithColumnStride(int N, int M) throws DMLRuntimeException {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		cudnnSetTensorNdDescriptor(tensorDescriptor, LibMatrixCUDA.CUDNN_DATA_TYPE, 3, new int[] {N, M, 1}, new int[] {M, 1, 1});
		return tensorDescriptor;
	}


	@Override
	public void close() {
		if(dropoutDesc != null)
			cudnnDestroyDropoutDescriptor(dropoutDesc);
		if(rnnDesc != null)
			cudnnDestroyRNNDescriptor(rnnDesc);
		if(hxDesc != null)
			cudnnDestroyTensorDescriptor(hxDesc);
		if(hyDesc != null)
			cudnnDestroyTensorDescriptor(hyDesc);
		if(cxDesc != null)
			cudnnDestroyTensorDescriptor(cxDesc);
		if(cyDesc != null)
			cudnnDestroyTensorDescriptor(cyDesc);
		if(wDesc != null)
			cudnnDestroyFilterDescriptor(wDesc);
		if(xDesc != null) {
			for(cudnnTensorDescriptor dsc : xDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
		}
		if(yDesc != null) {
			for(cudnnTensorDescriptor dsc : yDesc) {
				cudnnDestroyTensorDescriptor(dsc);
			}
		}
		if(sizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, workSpace);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}
		if(reserveSpaceSizeInBytes != 0) {
			try {
				gCtx.cudaFreeHelper(instName, reserveSpace);
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		}	
	}
}

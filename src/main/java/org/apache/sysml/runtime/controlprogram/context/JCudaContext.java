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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;
import jcuda.jcudnn.cudnnHandle;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcudnn.JCudnn.cudnnSetFilterNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

/**
 * Setup:
 * 1. Install CUDA 7.5
 * 2. Install CuDNN v4 from http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-win-x64-v4.0-prod.zip
 * 3. Download JCuda binaries version 0.7.5b and JCudnn version 0.7.5. Copy the DLLs into C:\lib (or /lib) directory. Link: http://www.jcuda.org/downloads/downloads.html
 *
 */
public class JCudaContext extends GPUContext {
	
	public cudnnHandle cudnnHandle;

	static {
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
	}
	
	public JCudaContext() {
		cudnnHandle = new cudnnHandle();
		cudnnCreate(cudnnHandle);
	}

	@Override
	public void destroy() {
		cudnnDestroy(cudnnHandle);
	}
	
	public Pointer pointerTo(double value) {
        return Pointer.to(new double[] { value });
    }
	
	public Pointer allocateDoubleArrayOnGPU(double[] data, int N, int C, int H, int W) {
		Pointer ret = new Pointer();
		cudaMalloc(ret, N * C * H * W * Sizeof.DOUBLE);
		cudaMemcpy(ret, Pointer.to(data), N * C * H * W * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		return ret;
	}
	
	public double [] getDoubleArrayFromDevice(Pointer pointer, int numElements) {
		double [] ret = new double[numElements];
		cudaMemcpy(Pointer.to(ret), pointer, numElements*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		return ret;
	}
	
	public void getDoubleArrayFromGPU(Pointer pointer, double [] ret) {
		cudaMemcpy(Pointer.to(ret), pointer, ret.length*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
	}
	
	public cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor ret = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(ret);
		cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		return ret;
	}
	
	public cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		int filterDim[] = { K, C, R, S };
		cudnnSetFilterNdDescriptor(filterDesc, CUDNN_DATA_DOUBLE, 4, filterDim);
		return filterDesc;
	}
	
}

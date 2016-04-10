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

import java.util.Collections;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

import java.util.Comparator;

import jcuda.Pointer;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;
import jcuda.jcudnn.cudnnHandle;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroy;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolutionNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilterNdDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.cudaError.cudaSuccess;

/**
 * Setup:
 * 1. Install CUDA 7.5
 * 2. Install CuDNN v4 from http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-win-x64-v4.0-prod.zip
 * 3. Download JCuda binaries version 0.7.5b and JCudnn version 0.7.5. Copy the DLLs into C:\lib (or /lib) directory. Link: http://www.jcuda.org/downloads/downloads.html
 *
 */
public class JCudaContext extends GPUContext {
	
	private static final Log LOG = LogFactory.getLog(JCudaContext.class.getName());
	
	public static boolean DEBUG = true;
	public cudnnHandle cudnnHandle;
	
	public static long totalNumBytes = 0;
	public static long availableNumBytesWithoutUtilFactor = 0;
	// Fraction of available memory to use. The available memory is computer when the JCudaContext is created
	// to handle the tradeoff on calling cudaMemGetInfo too often. 
	public static double GPU_MEMORY_UTILIZATION_FACTOR = 0.9; 
	public static boolean REFRESH_AVAILABLE_MEMORY_EVERY_TIME = true;
	
	static {
		JCuda.setExceptionsEnabled(true);
		JCudnn.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);
		cuInit(0); // Initialize the driver
		// Obtain the number of devices
        int deviceCountArray[] = { 0 };
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        LOG.info("Total number of GPUs on the machine: " + deviceCount);
	}
	
	public long getAvailableMemory() {
		if(REFRESH_AVAILABLE_MEMORY_EVERY_TIME) {
			long free [] = { 0 };
	        long total [] = { 0 };
	        if(cudaMemGetInfo(free, total) == cudaSuccess) {
	        	totalNumBytes = total[0];
	        	availableNumBytesWithoutUtilFactor = free[0];
	        }
	        else {
	        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
	        }
		}
		
		return (long) (availableNumBytesWithoutUtilFactor*GPU_MEMORY_UTILIZATION_FACTOR);
	}
	
	
	public JCudaContext() {
		cudnnHandle = new cudnnHandle();
		cudnnCreate(cudnnHandle);
		long free [] = { 0 };
        long total [] = { 0 };
        if(cudaMemGetInfo(free, total) == cudaSuccess) {
        	totalNumBytes = total[0];
        	availableNumBytesWithoutUtilFactor = free[0];
        }
        else {
        	throw new RuntimeException("ERROR: Unable to get memory information of the GPU.");
        }
        LOG.info("Total GPU memory: " + (totalNumBytes*(1e-6)) + " MB");
        LOG.info("Available GPU memory: " + (availableNumBytesWithoutUtilFactor*(1e-6)) + " MB");
	}

	@Override
	public void destroy() {
		cudnnDestroy(cudnnHandle);
	}
	
	public Pointer pointerTo(double value) {
        return Pointer.to(new double[] { value });
    }
	
//	public Pointer allocateDoubleArrayOnGPU(double[] data, int N, int C, int H, int W) {
//		Pointer ret = new Pointer();
//		cudaMalloc(ret, N * C * H * W * Sizeof.DOUBLE);
//		cudaMemcpy(ret, Pointer.to(data), N * C * H * W * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
//		return ret;
//	}
//	
//	public double [] getDoubleArrayFromDevice(Pointer pointer, int numElements) {
//		double [] ret = new double[numElements];
//		cudaMemcpy(Pointer.to(ret), pointer, numElements*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
//		return ret;
//	}
//	
//	public void getDoubleArrayFromGPU(Pointer pointer, double [] ret) {
//		cudaMemcpy(Pointer.to(ret), pointer, ret.length*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
//	}
	
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

	@Override
	public Object prepare(MatrixBlock mat, boolean isInput, boolean lock)
			throws DMLRuntimeException {
		if(mat.gpuPointer == null) {
			mat.gpuPointer = GPUPointer.createGPUPointer(mat, this);
			long GPUSize = mat.gpuPointer.getSizeOnDevice();
			
			// Ensure enough memory while allocating the matrix
			if(GPUSize > getAvailableMemory()) {
				if(DEBUG)
					LOG.info("There is not enough memory on device. Eviction is issued!");
				evict(GPUSize);
			}
			
			mat.gpuPointer.allocateMemoryOnDevice();
			synchronized(evictionLock) {
				allocatedPointers.add(mat.gpuPointer);
			}
			if (isInput) {
				mat.gpuPointer.copyFromHostToDevice();
			}
//			else {
//				mat.gpuPointer.copyFromHostToDevice();
//			}
		}
		mat.gpuPointer.isLocked = lock;
		return ((JCudaPointer)mat.gpuPointer).jcudaPointer; 
	}
	
	Boolean evictionLock = new Boolean(true);

	@Override
	public void unlock(MatrixBlock mat, boolean isGPUCopyModified) {
		mat.gpuPointer.isLocked = false;
		mat.gpuPointer.isDeviceCopyModified = isGPUCopyModified;
	}
	
	
	/**
	 * It finds matrix toBeRemoved such that toBeRemoved.GPUSize >= size
	 * // TODO: it is the smallest matrix size that satisfy the above condition. For now just evicting the largest pointer.
	 * Then returns toBeRemoved. 
	 * 
	 */
	protected void evict(long GPUSize) throws DMLRuntimeException {
		if(allocatedPointers.size() == 0) {
			throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
		}
		
		synchronized(evictionLock) {
			Collections.sort(allocatedPointers, new Comparator<GPUPointer>() {
	
				@Override
				public int compare(GPUPointer p1, GPUPointer p2) {
					if(p1.isLocked && p2.isLocked) {
						return 0;
					}
					else if(p1.isLocked && !p2.isLocked) {
						// p2 by default is considered larger
						return 1;
					}
					else if(!p1.isLocked && p2.isLocked) {
						return -1;
					}
					long p1Size = 0; long p2Size = 0;
					try {
						p1Size = p1.getSizeOnDevice();
						p2Size = p2.getSizeOnDevice();
					} catch (DMLRuntimeException e) {
						throw new RuntimeException(e);
					}
					if(p1Size == p2Size) {
						return 0;
					}
					else if(p1Size < p2Size) {
						return 1;
					}
					else {
						return -1;
					}
				}
			});
			
			
			while(GPUSize > getAvailableMemory() && allocatedPointers.size() > 0) {
				GPUPointer toBeRemoved = allocatedPointers.get(allocatedPointers.size() - 1);
				if(toBeRemoved.isLocked) {
					throw new DMLRuntimeException("There is not enough memory on device for this matrix!");
				}
				if(toBeRemoved.isDeviceCopyModified) {
					toBeRemoved.copyFromDeviceToHost();
				}
				remove(toBeRemoved.mat);
			}
		}
	}


	@Override
	public void remove(MatrixBlock mat) throws DMLRuntimeException {
		if(mat != null && mat.gpuPointer != null) {
			synchronized(evictionLock) {
				allocatedPointers.remove(mat.gpuPointer);
			}
			mat.gpuPointer.deallocateMemoryOnDevice();
		}
	}


	@Override
	public void conv2d(MatrixBlock image, MatrixBlock filter, MatrixBlock outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
			throws DMLRuntimeException {
		JCudaContext gpuCtx = (JCudaContext) this;
		cudnnTensorDescriptor srcTensorDesc = null;
		cudnnTensorDescriptor dstTensorDesc = null;
		cudnnFilterDescriptor filterDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		Pointer workSpace = null;
		long sizeInBytes = 0;
		Pointer alpha = null;
		Pointer beta = null;
		try {
			// Allocate descriptors
			srcTensorDesc = gpuCtx.allocateTensorDescriptor(N, C, H, W);
			dstTensorDesc = gpuCtx.allocateTensorDescriptor(N, K, P, Q);
			filterDesc = gpuCtx.allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			Pointer imagePointer = (Pointer) gpuCtx.prepare(image, true, true);
			Pointer filterPointer = (Pointer) gpuCtx.prepare(filter, true, true);
			Pointer dstPointer = (Pointer) gpuCtx.prepare(outputBlock, false, true);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			
			long sizeInBytesArray[] = { 0 };
            workSpace = new Pointer();
            cudnnGetConvolutionForwardWorkspaceSize(gpuCtx.cudnnHandle, 
                    srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
                    algo, sizeInBytesArray);
            
			alpha = gpuCtx.pointerTo(1.0); // TODO
			beta = gpuCtx.pointerTo(0.0f);
			int status = cudnnConvolutionForward(gpuCtx.cudnnHandle, alpha, 
					srcTensorDesc, imagePointer, 
					filterDesc, filterPointer,
					convDesc, algo, workSpace, sizeInBytes, beta,
					dstTensorDesc, dstPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			gpuCtx.unlock(image, false);
			gpuCtx.unlock(filter, false);
			gpuCtx.unlock(outputBlock, true);
		}
		finally {
			
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			
			if(srcTensorDesc != null)
				cudnnDestroyTensorDescriptor(srcTensorDesc);
			if(dstTensorDesc != null)
				cudnnDestroyTensorDescriptor(dstTensorDesc);
			if(filterDesc != null)
				cudnnDestroyFilterDescriptor(filterDesc);
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}
	
	cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		int upscale[] = { 1, 1 };
		cudnnSetConvolutionNdDescriptor(convDesc, 2, padding, strides, upscale, 
				CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
		return convDesc;
	}


	@Override
	public void conv2d_backward_filter(MatrixBlock image, MatrixBlock dout,
			MatrixBlock outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		
		JCudaContext gpuCtx = this;
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xTensorDesc = null;
		cudnnTensorDescriptor doutTensorDesc = null;
		cudnnFilterDescriptor dwDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			xTensorDesc = gpuCtx.allocateTensorDescriptor(N, C, H, W);
			doutTensorDesc = gpuCtx.allocateTensorDescriptor(N, K, P, Q);
			dwDesc = gpuCtx.allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			Pointer imagePointer = (Pointer) gpuCtx.prepare(image, true, true);
			Pointer doutPointer =  (Pointer) gpuCtx.prepare(dout, true, true);
			Pointer dwPointer = (Pointer) gpuCtx.prepare(outputBlock, false, true);
			
			alpha = gpuCtx.pointerTo(1.0); // TODO
			beta = gpuCtx.pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardFilterWorkspaceSize(gpuCtx.cudnnHandle,
					xTensorDesc, doutTensorDesc, convDesc, dwDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardFilter(gpuCtx.cudnnHandle, alpha, xTensorDesc, imagePointer, 
					doutTensorDesc, doutPointer, convDesc, algo, workSpace, sizeInBytes, beta, dwDesc, dwPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			
			gpuCtx.unlock(image, false);
			gpuCtx.unlock(dout, false);
			gpuCtx.unlock(outputBlock, true);
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(xTensorDesc != null)
				cudnnDestroyTensorDescriptor(xTensorDesc);
			if(doutTensorDesc != null)
				cudnnDestroyTensorDescriptor(doutTensorDesc);
			if(dwDesc != null)
				cudnnDestroyFilterDescriptor(dwDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
		
	}


	@Override
	public void copyDeviceToHost(MatrixBlock mat) throws DMLRuntimeException {
		mat.gpuPointer.copyFromDeviceToHost();
	}
	
}

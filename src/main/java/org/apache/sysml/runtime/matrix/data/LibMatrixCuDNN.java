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

import static jcuda.jcudnn.JCudnn.cudnnActivationForward;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetActivationDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardTraining;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardInference;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;
import static jcuda.runtime.JCuda.cudaMemset;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnStatus;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.LibMatrixCuMatMult.CuMatMultParameters;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN.PoolingType;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;

import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL;

/**
 * This class contains method that invoke CuDNN operations.
 */
public class LibMatrixCuDNN extends LibMatrixCUDA {

	// Currently we only use nnz information from the sparse matrix which is pre-computed
	// TODO: experiment how often does dense matrix is empty where recomputing nnz before calling CuDNN will help
	private static final boolean RECOMPUTE_DENSE_NNZ = false;
	
	protected static int CONVOLUTION_PREFERENCE = cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	private static final Log LOG = LogFactory.getLog(LibMatrixCuDNN.class.getName());

	protected static cudnnHandle getCudnnHandle(GPUContext gCtx) {
		return gCtx.getCudnnHandle();
	}
	
	/**
	 * Does a 2D convolution followed by a bias_add
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image    input image matrix object
	 * @param bias     bias matrix object
	 * @param filter   filter matrix object
	 * @param output   output matrix object
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param K        number of output "channels"
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void conv2dBiasAdd(GPUContext gCtx, String instName, MatrixObject image, MatrixObject bias, MatrixObject filter, MatrixObject output, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, double intermediateMemoryBudget) {
		conv2d(gCtx, instName, image, filter, output, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, intermediateMemoryBudget);
		//cudaDeviceSynchronize;
		biasAdd(gCtx, instName, output, bias, output);
	}
	
	/**
	 * Performs im2col operation on GPU
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input matrix object
	 * @param isSparseImage is input image sparse
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @return output im2col pointer (the caller is expected to free this pointer) or null if image is an empty matrix
	 */
	private static Pointer denseIm2col(GPUContext gCtx, String instName, MatrixObject image, boolean isSparseImage, long N, long C, long H, long W,
			int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q) {
		Pointer im2colPointer = null;
		long t1 = ConfigurationManager.isFinegrainedStatistics() ? System.nanoTime() : 0;
		if(isSparseImage) {
			CSRPointer inPointer = getSparsePointer(gCtx, image, instName);
			if(inPointer.nnz < 0) {
				throw new DMLRuntimeException("Unknown number of nonzeroes in denseIm2col");
			}
			else if(inPointer.nnz > 0) {
				im2colPointer = gCtx.allocate(instName, C*R*S*N*P*Q*sizeOfDataType);
				getCudaKernels(gCtx).launchKernel("sparse_dense_im2col", ExecutionConfig.getConfigForSimpleVectorOperations(toInt(inPointer.nnz)), 
						inPointer.val, inPointer.rowPtr, inPointer.colInd, im2colPointer, inPointer.nnz, N, 
						C*H*W, H*W, W, R, S, P, Q, P*Q, R*S, N*P*Q, stride_h, stride_w, pad_h, pad_w);
				if (ConfigurationManager.isFinegrainedStatistics())
					GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_IM2COL_KERNEL, System.nanoTime() - t1);
			}
			else
				return null;
		}
		else {
			im2colPointer = gCtx.allocate(instName, C*R*S*N*P*Q*sizeOfDataType);
			Pointer imagePointer = getDensePointerForCuDNN(gCtx, image, instName);
			getCudaKernels(gCtx).launchKernel("dense_dense_im2col", ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*C*H*W)), 
					imagePointer, im2colPointer, N*C*H*W, 
					C*H*W, H*W, W, R, S, P, Q, P*Q, R*S, N*P*Q, stride_h, stride_w, pad_h, pad_w);
			if (ConfigurationManager.isFinegrainedStatistics())
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_IM2COL_KERNEL, System.nanoTime() - t1);
		}
		return im2colPointer;
	}
	
	/**
	 * Performs a 2D convolution
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input matrix object
	 * @param filter filter matrix object
	 * @param outputBlock output matrix object
	 * @param N        number of input images
	 * @param C        number of channels
	 * @param H        height of each image
	 * @param W        width of each image
	 * @param K        number of output "channels"
	 * @param R        height of filter
	 * @param S        width of filter
	 * @param pad_h    padding height
	 * @param pad_w    padding width
	 * @param stride_h stride height
	 * @param stride_w string width
	 * @param P        output height
	 * @param Q        output width
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void conv2d(GPUContext gCtx, String instName, MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q, double intermediateMemoryBudget) {

		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;
		long NPQ = N*P*Q;
		
		boolean isSparseFilter = isInSparseFormat(gCtx, filter);
		long filterNnz = getNnz(gCtx, instName, filter, RECOMPUTE_DENSE_NNZ);
		if(filterNnz == 0) {
			return; // since filter is empty
		}
		boolean isSparseImage = isInSparseFormat(gCtx, image);
		long imageNnz = getNnz(gCtx, instName, image, RECOMPUTE_DENSE_NNZ);
		if(imageNnz == 0) {
			return; // since image is empty
		}
		Pointer dstPointer = getDensePointerForCuDNN(gCtx, outputBlock, instName);
		
		if(NCHW < maxNumElementsOfCuDNNTensor && NKPQ < maxNumElementsOfCuDNNTensor && KCRS < maxNumElementsOfCuDNNTensor) {
			if(isSparseFilter && 
				(OptimizerUtils.estimateSizeExactSparsity(CRS, NPQ, 1.0) + OptimizerUtils.estimateSizeExactSparsity(K, NPQ, 1.0)) < 
					Math.min(LibMatrixCuDNNConvolutionAlgorithm.MAX_WORKSPACE_LIMIT_BYTES, intermediateMemoryBudget)) {
				// Sparse filter conv2d
				// Perform dense im2col
				Pointer im2colPointer = denseIm2col(gCtx, instName, image, isSparseImage,
						N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w, P, Q);
				
				// Perform matrix multiplication
				CSRPointer filterPointer = filter.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
				Pointer matmultOutputPointer = gCtx.allocate(instName, NKPQ*sizeOfDataType);
				LibMatrixCuMatMult.sparseDenseMatMult(gCtx, instName, matmultOutputPointer, filterPointer, im2colPointer, K, CRS, CRS, NPQ, K, NPQ, false, false);
				gCtx.cudaFreeHelper(instName, im2colPointer, gCtx.EAGER_CUDA_FREE);
				
				// Perform reorg_knpq a reorg operation of matmultOutputPointer matrix with dimensions [K, NPQ]
				// and return a matrix dstPointer with dimensions [N, KPQ]
				long t1 = ConfigurationManager.isFinegrainedStatistics() ? System.nanoTime() : 0;
				getCudaKernels(gCtx).launchKernel("reorg_knpq", ExecutionConfig.getConfigForSimpleVectorOperations(toInt(NKPQ)), 
						matmultOutputPointer, dstPointer, NKPQ, NPQ, KPQ, P*Q);
				if (ConfigurationManager.isFinegrainedStatistics())
					GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_DENSE_REORG_KNPQ_KERNEL, System.nanoTime() - t1);
				gCtx.cudaFreeHelper(instName, matmultOutputPointer, gCtx.EAGER_CUDA_FREE);
			}
			else {
				// Filter and output are accounted as dense in the memory estimation for conv2d
				double overhead = isSparseFilter ? OptimizerUtils.estimateSizeExactSparsity(K, CRS, 1.0) : 0;
				overhead += isSparseImage ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;

				Pointer filterPointer = getDensePointerForCuDNN(gCtx, filter, instName);
				
				// Required for LibMatrixCuDNNConvolutionAlgorithm
				long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
				int localN = overhead <= intermediateMemoryBudget ? N : 1;
				
				try(LibMatrixCuDNNConvolutionAlgorithm algo = 
						LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionForwardAlgorithm(gCtx, instName, 
						localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
					if(localN == N) {
						// Perform all-input all-channel conv2d
						Pointer imagePointer = getDensePointerForCuDNN(gCtx, image, instName);
						cudnnConv2d(gCtx, instName, imagePointer, filterPointer, dstPointer, algo);
					}
					else {
						try(LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image)) {
							for(int n = 0; n < N; n++) {
								// Perform one-input all-channel conv2d
								cudnnConv2d(gCtx, instName, imgFetcher.getNthRow(n), filterPointer, dstPointer.withByteOffset(n*KPQ*sizeOfDataType), algo);
							}
						}
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, K, CRS, N, KPQ);
		}
	}
	
	/**
	 * Performs an "softmax" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void softmax(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : softmax" + ", GPUContext=" + gCtx);
		}
		cudnnTensorDescriptor tensorDesc = allocateTensorDescriptor(toInt(in1.getNumRows()), toInt(in1.getNumColumns()), 1, 1);
		Pointer srcPointer = getDensePointerForCuDNN(gCtx, in1, instName);
		MatrixObject out = ec.getMatrixObject(outputName);
		ec.allocateGPUMatrixObject(outputName, in1.getNumRows(), in1.getNumColumns());
		out.getGPUObject(gCtx).allocateAndFillDense(0);
		Pointer dstPointer = getDensePointerForCuDNN(gCtx, out, instName);
		JCudnn.cudnnSoftmaxForward(gCtx.getCudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, 
                one(), tensorDesc, srcPointer,
                zero(), tensorDesc, dstPointer);
		cudnnDestroyTensorDescriptor(tensorDesc);
	}
	
	/**
	 * Convenience method to get tensor descriptor
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @return cudnn tensor descriptor
	 */
	private static cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(tensorDescriptor);
		cudnnSetTensor4dDescriptor(tensorDescriptor, CUDNN_TENSOR_NCHW, LibMatrixCUDA.CUDNN_DATA_TYPE, N, C, H, W);
		return tensorDescriptor;
	}


	/**
	 * Throw an user-friendly error that shows limitation of invoking a cuDNN kernel
	 *  
	 * @param dim1 input1 number of rows
	 * @param dim2 input1 number of columns
	 * @param dim3 input2 number of rows
	 * @param dim4 input2 number of columns
	 * @param dim5 output number of rows
	 * @param dim6 output number of columns
	 */
	private static void throwCuDNNDimensionError(long dim1, long dim2, long dim3, long dim4) {
		throw new DMLRuntimeException("The dimensions of input/output matrices is too large to execute a CuDNN kernel. "
				+ "Max CuDNN matrix size:" + maxNumElementsOfCuDNNTensor + ". "
				+ "Given input matrix dimensions: [" + dim1 + "," + dim2 + "]. Output dimension:  [" + dim3 + "," + dim4 + "].");
	}

	/**
	 * Throw an user-friendly error that shows limitation of invoking a cuDNN kernel
	 *  
	 * @param dim1 input1 number of rows
	 * @param dim2 input1 number of columns
	 * @param dim3 input2 number of rows
	 * @param dim4 input2 number of columns
	 * @param dim5 output number of rows
	 * @param dim6 output number of columns
	 */
	private static void throwCuDNNDimensionError(long dim1, long dim2, long dim3, long dim4, long dim5, long dim6) {
		throw new DMLRuntimeException("The dimensions of input/output matrices is too large to execute a CuDNN kernel. "
				+ "Max CuDNN matrix size:" + maxNumElementsOfCuDNNTensor + ". "
				+ "Given input matrix dimensions: [" + dim1 + "," + dim2 + "], [" + dim3 + "," + dim4 + "]. Output dimension: [" + dim5 + "," + dim6 + "]");
	}

	/**
	 * Performs 2D convolution
	 * Takes up an insignificant amount of intermediate space when CONVOLUTION_PREFERENCE is set to CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
	 * Intermediate space is required by the filter descriptor and convolution descriptor which are metadata structures and don't scale with the size of the input
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image    the input matrix (or image) allocated on the GPU
	 * @param filter   the filter allocated on the GPU
	 * @param output   the output matrix allocated on the GPU
	 * @param algo     cudnn algorithm wrapper
	 */
	private static void cudnnConv2d(GPUContext gCtx, String instName, Pointer image, Pointer filter, Pointer output, 
			LibMatrixCuDNNConvolutionAlgorithm algo) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2d" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = 0;
			if (ConfigurationManager.isFinegrainedStatistics()) t1 = System.nanoTime();
			int status = cudnnConvolutionForward(getCudnnHandle(gCtx), one(),
					algo.nchwTensorDesc, image,
					algo.filterDesc, filter,
					algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(),
					algo.nkpqTensorDesc, output);
			if (ConfigurationManager.isFinegrainedStatistics())
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_FORWARD_LIB, System.nanoTime() - t1);
			if (status != cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		} 
	}

	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input image
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @param K number of filters
	 * @param R filter height
	 * @param S filter width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param P output activation height
	 * @param Q output activation width
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void conv2dBackwardFilter(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) {
		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;
		
		boolean isSparseDout = isInSparseFormat(gCtx, dout);
		long doutNnz = getNnz(gCtx, instName, dout, RECOMPUTE_DENSE_NNZ);
		if(doutNnz == 0) {
			return; // since dout is empty
		}
		boolean isSparseImage = isInSparseFormat(gCtx, image);
		long imageNnz = getNnz(gCtx, instName, image, RECOMPUTE_DENSE_NNZ);
		if(imageNnz == 0) {
			return; // since image is empty
		}
		
		if(NCHW < maxNumElementsOfCuDNNTensor && NKPQ < maxNumElementsOfCuDNNTensor && KCRS < maxNumElementsOfCuDNNTensor) {
			Pointer dwPointer = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			double overhead = isSparseImage ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			overhead += isSparseDout ? OptimizerUtils.estimateSizeExactSparsity(N, KPQ, 1.0) : 0;

			// Required for LibMatrixCuDNNConvolutionAlgorithm
			long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
			int localN = overhead <= intermediateMemoryBudget ? N : 1;
			
			try(LibMatrixCuDNNConvolutionAlgorithm algo = 
					LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionBackwardFilterAlgorithm(gCtx, instName, 
					localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
				if(localN == N) {
					// Perform all-input all-channel conv2dBackwardFilter
					Pointer imagePointer = getDensePointerForCuDNN(gCtx, image, instName);
					Pointer doutPointer = getDensePointerForCuDNN(gCtx, dout, instName);
					cudnnConv2dBackwardFilter(gCtx, instName, imagePointer, doutPointer, dwPointer, algo);
				}
				else {
					try(LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
						LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout)) {
						// Perform one-input conv2dBackwardFilter
						Pointer tempdwPointer = gCtx.allocate(instName, KCRS*sizeOfDataType);
						for(int n = 0; n < N; n++) {
							long t0 = ConfigurationManager.isFinegrainedStatistics() ? System.nanoTime() : 0;
							cudaMemset(tempdwPointer, 0, KCRS*sizeOfDataType);
							if(ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SET_ZERO, System.nanoTime() - t0);
							// Perform one-input conv2dBackwardFilter
							cudnnConv2dBackwardFilter(gCtx, instName, imgFetcher.getNthRow(n), doutFetcher.getNthRow(n), tempdwPointer, algo);
							getCudaKernels(gCtx).launchKernel("inplace_add",
									ExecutionConfig.getConfigForSimpleMatrixOperations(K, toInt(CRS)),
									tempdwPointer, dwPointer, K, toInt(CRS));

						}
						// Deallocate temporary array to hold one element of input
						gCtx.cudaFreeHelper(instName, tempdwPointer, true);
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, KPQ, K, CRS);
		}
	}

	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param imagePointer pointer to input image
	 * @param doutPointer pointer to errors from next layer
	 * @param dwPointer  output errors
	 * @param algo     cudnn algorithm wrapper
	 */
	private static void cudnnConv2dBackwardFilter(GPUContext gCtx, String instName, Pointer imagePointer, Pointer doutPointer,
			Pointer dwPointer, LibMatrixCuDNNConvolutionAlgorithm algo) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2dBackwardFilter" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = ConfigurationManager.isFinegrainedStatistics() ? System.nanoTime() : 0;
			int status = cudnnConvolutionBackwardFilter(getCudnnHandle(gCtx), one(), algo.nchwTensorDesc, imagePointer,
					algo.nkpqTensorDesc, doutPointer, algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(), algo.filterDesc, dwPointer);
			if (ConfigurationManager.isFinegrainedStatistics())
				GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_FILTER_LIB, System.nanoTime() - t1);
			if (status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		} 
	}

	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param filter filter used in conv2d
	 * @param dout errors from next layer
	 * @param output  output errors
	 * @param N number of images
	 * @param C number of channels
	 * @param H height
	 * @param W width
	 * @param K number of filters
	 * @param R filter height
	 * @param S filter width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param P output activation height
	 * @param Q output activation width
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void conv2dBackwardData(GPUContext gCtx, String instName, MatrixObject filter, MatrixObject dout,
			MatrixObject output, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, double intermediateMemoryBudget) {
		long CHW = C*H*W; long KPQ = K*P*Q; long CRS = C*R*S; 
		long NCHW = N*CHW; long NKPQ = N*KPQ; long KCRS = K*CRS;

		boolean isSparseFilter = isInSparseFormat(gCtx, filter);
		long filterNnz = getNnz(gCtx, instName, filter, RECOMPUTE_DENSE_NNZ);
		if(filterNnz == 0) {
			return; // since filter is empty
		}
		boolean isSparseDout = isInSparseFormat(gCtx, dout);
		long doutNnz = getNnz(gCtx, instName, dout, RECOMPUTE_DENSE_NNZ);
		if(doutNnz == 0) {
			return; // since dout is empty
		}
		
		if(NCHW < maxNumElementsOfCuDNNTensor && NKPQ < maxNumElementsOfCuDNNTensor && KCRS < maxNumElementsOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			double overhead = isSparseFilter ? OptimizerUtils.estimateSizeExactSparsity(K, CRS, 1.0) : 0;
			overhead += isSparseDout ? OptimizerUtils.estimateSizeExactSparsity(N, KPQ, 1.0) : 0;
			Pointer filterPointer = getDensePointerForCuDNN(gCtx, filter, instName);
			Pointer dstPointer = getDensePointerForCuDNN(gCtx, output, instName);
			
			// Required for LibMatrixCuDNNConvolutionAlgorithm
			long workspaceLimit = (long) (intermediateMemoryBudget-overhead);
			int localN = overhead <= intermediateMemoryBudget ? N : 1;
			
			try(LibMatrixCuDNNConvolutionAlgorithm algo = 
					LibMatrixCuDNNConvolutionAlgorithm.cudnnGetConvolutionBackwardDataAlgorithm(gCtx, instName, 
					localN, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, workspaceLimit)) {
				if(localN == N) {
					// Perform all-input all-channel conv2dBackwardData
					Pointer doutPointer = getDensePointerForCuDNN(gCtx, dout, instName);
					cudnnConv2dBackwardData(gCtx, instName, filterPointer, doutPointer, dstPointer, algo);
				}
				else {
					try(LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout)) {
						for(int n = 0; n < N; n++) {
							cudnnConv2dBackwardData(gCtx, instName, doutFetcher.getNthRow(n), filterPointer, dstPointer.withByteOffset(n*CHW*sizeOfDataType), algo);
						}
					}
				}
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, KPQ, K, CRS);
		}
	}

	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param w pointer to filter used in conv2d
	 * @param dy pointer to errors from next layer
	 * @param dx pointer to  output errors
	 * @param algo cudnn algorithm wrapper
	 */
	private static void cudnnConv2dBackwardData(GPUContext gCtx, String instName, Pointer w, Pointer dy,
			Pointer dx, LibMatrixCuDNNConvolutionAlgorithm algo) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : conv2dBackwardData" + ", GPUContext=" + gCtx);
		}
		try {
			long t1 = ConfigurationManager.isFinegrainedStatistics() ? System.nanoTime() : 0;
			int status = cudnnConvolutionBackwardData(getCudnnHandle(gCtx), one(), algo.filterDesc, w,
					algo.nkpqTensorDesc, dy, algo.convDesc, algo.algo, algo.workSpace, algo.sizeInBytes, zero(), algo.nchwTensorDesc, dx);
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CONVOLUTION_BACKWARD_DATA_LIB, System.nanoTime() - t1);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}	
	}

	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image image as matrix object
	 * @param outputBlock output matrix
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @param poolingType	type of pooling
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void pooling(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType, double intermediateMemoryBudget) {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		if(NCHW < maxNumElementsOfCuDNNTensor && NCPQ < maxNumElementsOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			long overhead = isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			Pointer y = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			if(overhead <= intermediateMemoryBudget) {
				Pointer x = getDensePointerForCuDNN(gCtx, image, instName);
				cudnnPoolingHelper(gCtx, instName, x, y, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
			}
			else {
				LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
				for(int n = 0; n < N; n++) {
					cudnnPoolingHelper(gCtx, instName, imgFetcher.getNthRow(n), y.withByteOffset(n*CPQ*sizeOfDataType), 1, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
				}
				imgFetcher.close();
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}
	
	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param x image as pointer
	 * @param outputBlock output matrix
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @param poolingType	type of pooling
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void pooling(GPUContext gCtx, String instName, Pointer x,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType, double intermediateMemoryBudget) {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		if(NCHW < maxNumElementsOfCuDNNTensor && NCPQ < maxNumElementsOfCuDNNTensor) {
			Pointer y = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			cudnnPoolingHelper(gCtx, instName, x, y, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}

	private static void cudnnPoolingHelper(GPUContext gCtx, String instName, Pointer x,
			Pointer y, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : perform pooling" + ", GPUContext=" + gCtx);
		}

		try(LibMatrixCuDNNPoolingDescriptors desc = 
				LibMatrixCuDNNPoolingDescriptors.cudnnPoolingDescriptors(gCtx, instName, N, C, H, W, K, R, S, 
						pad_h, pad_w, stride_h, stride_w, P, Q, poolingType)) {
			long t1=0,t2=0;
			if (ConfigurationManager.isFinegrainedStatistics()) t1 = System.nanoTime();
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
			if (ConfigurationManager.isFinegrainedStatistics()) t2 = System.nanoTime();
			int status = cudnnPoolingForward(getCudnnHandle(gCtx), desc.poolingDesc, one(), desc.xDesc, x, zero(), desc.yDesc, y);
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}
	}
	
	/**
	 * Performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image image as matrix object
	 * @param dout			delta matrix, output of previous layer
	 * @param maxpoolOutput (optional and can be null) output of maxpool forward function
	 * @param outputBlock output matrix
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @param poolingType	type of pooling
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void poolingBackward(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject maxpoolOutput, MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType, double intermediateMemoryBudget) {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		final boolean isMaxPoolOutputProvided = maxpoolOutput != null;
		
		if(NCHW < maxNumElementsOfCuDNNTensor && NCPQ < maxNumElementsOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			long overhead = isInSparseFormat(gCtx, image) ? OptimizerUtils.estimateSizeExactSparsity(N, CHW, 1.0) : 0;
			overhead += isInSparseFormat(gCtx, dout) ? OptimizerUtils.estimateSizeExactSparsity(N, CPQ, 1.0) : 0;
			Pointer dx = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			if(overhead <= intermediateMemoryBudget) {
				Pointer x = getDensePointerForCuDNN(gCtx, image, instName);
				Pointer dy = getDensePointerForCuDNN(gCtx, dout, instName);
				Pointer y = isMaxPoolOutputProvided ? getDensePointerForCuDNN(gCtx, maxpoolOutput, instName) : null;
				cudnnPoolingBackwardHelper(gCtx, instName, x, dy, y, dx, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
			}
			else {
				LibMatrixCuDNNInputRowFetcher imgFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, image);
				LibMatrixCuDNNInputRowFetcher doutFetcher = new LibMatrixCuDNNInputRowFetcher(gCtx, instName, dout);
				LibMatrixCuDNNInputRowFetcher maxPoolOutFetcher = isMaxPoolOutputProvided ? new LibMatrixCuDNNInputRowFetcher(gCtx, instName, maxpoolOutput) : null;
				for(int n = 0; n < N; n++) {
					Pointer x = imgFetcher.getNthRow(n);
					Pointer dy = doutFetcher.getNthRow(n);
					Pointer y = isMaxPoolOutputProvided ? maxPoolOutFetcher.getNthRow(n) : null;
					cudnnPoolingBackwardHelper(gCtx, instName, x, dy, y, 
							dx.withByteOffset(n*CHW*sizeOfDataType), 
							1, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
				}
				// Deallocate temporary array to hold one element of input
				imgFetcher.close();
				doutFetcher.close();
				if(isMaxPoolOutputProvided)
					maxPoolOutFetcher.close();
			}
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}
	
	/**
	 * Performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * This method computes the backpropogation errors for previous layer of maxpooling operation
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param x image as dense pointer
	 * @param dout			delta matrix, output of previous layer
	 * @param maxpoolOutput (optional and can be null) output of maxpool forward function
	 * @param outputBlock output matrix
	 * @param N				batch size
	 * @param C				number of channels
	 * @param H				height of image
	 * @param W				width of image
	 * @param K				number of filters
	 * @param R				height of filter
	 * @param S				width of filter
	 * @param pad_h			vertical padding
	 * @param pad_w			horizontal padding
	 * @param stride_h		horizontal stride
	 * @param stride_w		vertical stride
	 * @param P				(H - R + 1 + 2*pad_h)/stride_h
	 * @param Q				(W - S + 1 + 2*pad_w)/stride_w
	 * @param poolingType	type of pooling
	 * @param intermediateMemoryBudget intermediate memory budget
	 */
	public static void poolingBackward(GPUContext gCtx, String instName, Pointer x, MatrixObject dout,
			MatrixObject maxpoolOutput, MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType, double intermediateMemoryBudget) {
		long CHW = C*H*W; long CPQ = C*P*Q;  
		long NCHW = N*CHW; long NCPQ = N*CPQ; 

		final boolean isMaxPoolOutputProvided = maxpoolOutput != null;
		
		if(NCHW < maxNumElementsOfCuDNNTensor && NCPQ < maxNumElementsOfCuDNNTensor) {
			// Filter and output are accounted as dense in the memory estimation for conv2dBackwardData
			Pointer dx = getDensePointerForCuDNN(gCtx, outputBlock, instName);
			Pointer dy = getDensePointerForCuDNN(gCtx, dout, instName);
			Pointer y = isMaxPoolOutputProvided ? getDensePointerForCuDNN(gCtx, maxpoolOutput, instName) : null;
			cudnnPoolingBackwardHelper(gCtx, instName, x, dy, y, dx, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, P, Q, poolingType);
			
		}
		else {
			throwCuDNNDimensionError(N, CHW, N, CPQ);
		}
	}
	
	private static void cudnnPoolingBackwardHelper(GPUContext gCtx, String instName, 
			Pointer x, Pointer dy, Pointer y, Pointer dx, 
			int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q, PoolingType poolingType) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : maxpoolingBackward" + ", GPUContext=" + gCtx);
		}
		
		boolean isMaxPoolOutputProvided = (y != null);

		try(LibMatrixCuDNNPoolingDescriptors desc = 
				LibMatrixCuDNNPoolingDescriptors.cudnnPoolingBackwardDescriptors(gCtx, instName, N, C, H, W, K, R, S, 
						pad_h, pad_w, stride_h, stride_w, P, Q, poolingType)) {
			long t1=0, t2=0, t3=0;
			int status;
			if(!isMaxPoolOutputProvided) {
				if (ConfigurationManager.isFinegrainedStatistics()) t1 = System.nanoTime();
				long numBytes = N*C*P*Q*sizeOfDataType;
				y = gCtx.allocate(instName, numBytes);
				if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_INIT, System.nanoTime() - t1);
				if (ConfigurationManager.isFinegrainedStatistics()) t2 = System.nanoTime();
				status = cudnnPoolingForward(getCudnnHandle(gCtx), desc.poolingDesc, one(), desc.xDesc, x, zero(), desc.yDesc, y);
				if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_FORWARD_LIB, System.nanoTime() - t2);
				if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
					throw new DMLRuntimeException("Could not executed cudnnPoolingForward before cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
				}
			}
			if (ConfigurationManager.isFinegrainedStatistics()) t3 = System.nanoTime();
			status = cudnnPoolingBackward(getCudnnHandle(gCtx), desc.poolingDesc, one(), desc.yDesc, y, desc.dyDesc, dy, desc.xDesc, x, zero(), desc.dxDesc, dx);
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_MAXPOOLING_BACKWARD_LIB, System.nanoTime() - t3);

			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}
		finally {
			long t4=0;
			if (ConfigurationManager.isFinegrainedStatistics()) t4 = System.nanoTime();
			if(!isMaxPoolOutputProvided)
				gCtx.cudaFreeHelper(instName, y, gCtx.EAGER_CUDA_FREE);
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t4);
		}
	}

	private static void cudnnReLU(GPUContext gCtx, String instName, MatrixObject in, Pointer dstData, cudnnTensorDescriptor srcTensorDesc) {
		long t0=0;
		try {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : performCuDNNReLU" + ", GPUContext=" + gCtx);
			}
			cudnnTensorDescriptor dstTensorDesc = srcTensorDesc;

			Pointer srcData = getDensePointerForCuDNN(gCtx, in, instName);
			cudnnActivationDescriptor activationDescriptor = new cudnnActivationDescriptor();
			cudnnCreateActivationDescriptor(activationDescriptor);
			double dummy = -1;
			cudnnSetActivationDescriptor(activationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, dummy);
			if (ConfigurationManager.isFinegrainedStatistics()) t0 = System.nanoTime();
			cudnnActivationForward(getCudnnHandle(gCtx), activationDescriptor,
					one(), srcTensorDesc, srcData,
					zero(), dstTensorDesc, dstData);
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_ACTIVATION_FORWARD_LIB, System.nanoTime() - t0);
		} catch (CudaException e) {
			throw new DMLRuntimeException("Error in conv2d in GPUContext " + gCtx.toString() + " from Thread " + Thread.currentThread().toString(), e);
		}
		finally {
			long t1=0;
			if (ConfigurationManager.isFinegrainedStatistics()) t1 = System.nanoTime();
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_CUDNN_CLEANUP, System.nanoTime() - t1);
		}
	}

	/**
	 * Performs the relu operation on the GPU.
	 * @param ec currently active {@link ExecutionContext}
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName	name of the output matrix
	 */
	public static void relu(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		long N = in.getNumRows();
		long CHW = in.getNumColumns();
		Pointer dstData = getDenseOutputPointer(ec, gCtx, instName, outputName, in.getNumRows(), in.getNumColumns());
		long t0=0;
		if(N*CHW >= maxNumElementsOfCuDNNTensor) {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : relu custom kernel" + ", GPUContext=" + gCtx);
			}
			// Invokes relu(double* A,  double* ret, int rlen, int clen)
			Pointer srcData = getDensePointerForCuDNN(gCtx, in, instName); // TODO: FIXME: Add sparse kernel support for relu
			if (ConfigurationManager.isFinegrainedStatistics()) t0 = System.nanoTime();
			getCudaKernels(gCtx).launchKernel("relu",
					ExecutionConfig.getConfigForSimpleMatrixOperations(toInt(N), toInt(CHW)),
					srcData, dstData, toInt(N), toInt(CHW));
			if (ConfigurationManager.isFinegrainedStatistics()) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_RELU_KERNEL, System.nanoTime() - t0);
		}
		else {
			cudnnTensorDescriptor tensorDescriptor = new cudnnTensorDescriptor();
			cudnnCreateTensorDescriptor(tensorDescriptor);
			cudnnSetTensor4dDescriptor(tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_TYPE, toInt(N), 1, 1, toInt(CHW));
			cudnnReLU(gCtx, instName, in, dstData, tensorDescriptor);
			cudnnDestroyTensorDescriptor(tensorDescriptor);
		}
	}
	
	public static Pointer getDenseInputPointer(ExecutionContext ec, GPUContext gCtx, String instName, String inputName,
			long numRows, long numCols) throws DMLRuntimeException {
		MatrixObject output = ec.getMatrixInputForGPUInstruction(inputName, instName);
		return LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, output, instName, numRows, numCols);
	}
	
	public static Pointer getDenseInputPointer(ExecutionContext ec, GPUContext gCtx, String instName, String inputName) throws DMLRuntimeException {
		MatrixObject output = ec.getMatrixInputForGPUInstruction(inputName, instName);
		return LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, output, instName);
	}
	
	public static Pointer getDenseOutputPointer(ExecutionContext ec, GPUContext gCtx, String instName, String outputName,
			long numRows, long numCols) throws DMLRuntimeException {
		MatrixObject output = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, numRows, numCols); // Allocated the dense output matrix
		return getDensePointerForCuDNN(gCtx, output, instName, numRows, numCols);
	}
	
	public static Pair<MatrixObject, Pointer> getTemporaryCacheDensePointer(ExecutionContext ec, GPUContext gCtx, String instName, String varName,
			String scopeVarName, long sizeInBytes) throws DMLRuntimeException {
		int numRows = LibMatrixCUDA.toInt((long)Math.ceil( ((double) sizeInBytes) / sizeOfDataType ));;
		MatrixObject mo = null;
		if(ec.containsTemporaryCacheMatrix(varName, scopeVarName)) {
			mo = ec.getTemporaryCacheMatrixObject(varName, scopeVarName);  
		}
		else {
			int blocksize = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc = new MatrixCharacteristics(numRows, 1, blocksize, blocksize);
			MetaDataFormat meta = new MetaDataFormat(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			mo = new MatrixObject(ValueType.DOUBLE, OptimizerUtils.getUniqueTempFileName(), meta);
			mo.acquireModify(new MatrixBlock(numRows, 1, false));
			mo.release();
			// Donot set the matrix object in the symbol table
			// ec.setVariable(varName, mo);
			ec.getMatrixObject(scopeVarName).getTemporaryCacheData().put(varName, mo);
		}
		
		ec.getDenseMatrixOutputForGPUInstruction(mo, numRows, 1); // Allocated the dense output matrix
		Pointer ptr = getDensePointerForCuDNN(gCtx, mo, instName, numRows, 1);
		return new Pair<MatrixObject, Pointer>(mo, ptr);
	}
	
	public static void nnLstmBackward(ExecutionContext ec, GPUContext gCtx, String instName,
			Pointer X, Pointer out0, Pointer c0, MatrixObject W, Pointer dout, Pointer dc,  // input
			Pointer cache_out, Pointer cache_c, Pointer cache_ifog,
			Pointer dX, Pointer dW, Pointer db, Pointer dout0, Pointer dc0,  	// output
			boolean return_sequences, long N, long M, long D, long T) throws DMLRuntimeException {
		Pointer input = gCtx.allocate(instName, N*(D+M)*sizeOfDataType); 
		Pointer difog_raw = gCtx.allocate(instName, N*4*M*sizeOfDataType);
		Pointer dct = copy(gCtx, instName, dc, N*M);
		Pointer dinput = gCtx.allocate(instName, N*(D+M)*sizeOfDataType); // (N, D+M)
		Pointer tmpDb = gCtx.allocate(instName, 4*M*sizeOfDataType); // (1, 4M)
		
		// dW = dW + t(input) %*% difog_raw  # shape (D+M, 4M)
		CuMatMultParameters param1 = new CuMatMultParameters(N, D+M,
				N, 4*M, true, false, one(), one());
		
		// dinput = difog_raw %*% t(W)  # shape (N, D+M)
		CuMatMultParameters param2 = new CuMatMultParameters(N, 4*M,
				D+M, 4*M, false, true);
		
		CSRPointer wSparsePointer = null;
		Pointer wDensePointer = null;
		
		// TODO: Only dense weight supported for now
		boolean isWSparse = false; // isInSparseFormat(gCtx, W);
		if(isWSparse)
			wSparsePointer = W.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
		else
			wDensePointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, W, instName, D+M, 4*M);
		
		Pointer dout_t = return_sequences ? gCtx.allocate(instName, N*M*sizeOfDataType) : copy(gCtx, instName, dout, N*M);
		if(return_sequences) {
			getCudaKernels(gCtx).launchKernel("initializeDoutWhenReturnSeq",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*M)),
					dout, dout_t, T-1, toInt(M), toInt(T*M), toInt(N*M));
		}
		
		for(int t = toInt(T); t >= 1; t--) {
			// if (t == 1) { out_prev = out0; } else { out_prev = matrix(cache_out[t-1,], rows=N, cols=M) }
			Pointer out_prev = (t == 1) ? out0 : cache_out.withByteOffset((t-2)*N*M*sizeOfDataType); // since read-only
			
			// X_t = X[,(t-1)*D+1:t*D]  # shape (N, D)
			// input = cbind(X_t, out_prev)  # shape (N, D+M)
			getCudaKernels(gCtx).launchKernel("prepareInputNNLstm",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*(D+M))),
					X, out_prev, input, (t-1), toInt(M), toInt(D), toInt(T*D), toInt(D+M), toInt(N*(D+M)));
			
			// ct = matrix(cache_c[t,], rows=N, cols=M)  # shape (N, M)
			Pointer ct = cache_c.withByteOffset((t-1)*N*M*sizeOfDataType); // since read-only
			
			// ifog = matrix(cache_ifog[t,], rows=N, cols=4*M)
			Pointer ifog = cache_ifog.withByteOffset((t-1)*N*4*M*sizeOfDataType); // since read-only
			
			// i = ifog[,1:M]  # input gate, shape (N, M)
			// f = ifog[,M+1:2*M]  # forget gate, shape (N, M)
			// o = ifog[,2*M+1:3*M]  # output gate, shape (N, M)
			// g = ifog[,3*M+1:4*M]  # g gate, shape (N, M)
			// dct = dct + o*tanh::backward(dout_t, ct)  # shape (N, M)
			// do = tanh::forward(ct) * dout_t  # output gate, shape (N, M)
			// df = c_prev * dct  # forget gate, shape (N, M)
			// dc_prev = f * dct  # shape (N, M)
			// di = g * dct  # input gate, shape (N, M)
			// dg = i * dct  # g gate, shape (N, M)
			// di_raw = i * (1-i) * di
			// df_raw = f * (1-f) * df
			// do_raw = o * (1-o) * do
			// dg_raw = (1-g^2) * dg
			// difog_raw = cbind(di_raw, df_raw, do_raw, dg_raw)  # shape (N, 4M)
			getCudaKernels(gCtx).launchKernel("computeDifog_raw",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*M)),
					ifog, ct, dout_t, cache_c, c0, 
					difog_raw, dct, dc0, // output
					return_sequences ? 1 : 0, t-1, toInt(T), toInt(M), toInt(N*M));
			
			// dW = dW + t(input) %*% difog_raw  # shape (D+M, 4M)
			LibMatrixCuMatMult.denseDenseMatMult(gCtx.getCublasHandle(), instName, dW, input, difog_raw, param1);
			
			// dinput = difog_raw %*% t(W)  # shape (N, D+M)
			if(isWSparse) {
				if(wSparsePointer.nnz == 0) {
					cudaMemset(dinput, 0, N*(D+M)*sizeOfDataType);
				}
				else {
					LibMatrixCuMatMult.denseSparseMatMult(gCtx.getCusparseHandle(), instName, dinput, difog_raw, wSparsePointer, param2);
				}
			}
			else
				LibMatrixCuMatMult.denseDenseMatMult(gCtx.getCublasHandle(), instName, dinput, difog_raw, wDensePointer, param2);
			
			// db = db + colSums(difog_raw)  # shape (1, 4M)
			reduceCol(gCtx, instName, "reduce_col_sum", difog_raw, tmpDb, 1, toInt(4*M));
			matrixMatrixOp(gCtx, instName, tmpDb, db, 1, toInt(4*M), VectorShape.NONE.code(), VectorShape.NONE.code(), db, 
					new BinaryOperator(Plus.getPlusFnObject()));
			
			// jcuda.runtime.JCuda.cudaDeviceSynchronize();
			
			int size = toInt(Math.max(N*D, N*M));
			getCudaKernels(gCtx).launchKernel("postProcessNNLstmBackward",
					ExecutionConfig.getConfigForSimpleVectorOperations(size),
					dinput, dout0, dout, dout_t, dX, return_sequences ? 1 : 0, t-1, N, D, M, 
					toInt(N*D), toInt(N*M), toInt(T*D), toInt(T*M), toInt(D+M), size);
			
		}
		
		gCtx.cudaFreeHelper(instName, dout_t, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, input, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, difog_raw, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, dct, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, dinput, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, tmpDb, gCtx.EAGER_CUDA_FREE);
		
	}
	
	public static void nnLstm(ExecutionContext ec, GPUContext gCtx, String instName,
			Pointer X,  MatrixObject W, Pointer b, Pointer out0, Pointer c0, boolean return_sequences,
			Pointer out, Pointer c,  // output matrices
			Pointer cache_out, Pointer cache_c, Pointer cache_ifog, // temporary workspace passed to the backward function
			long N, long M, long D, long T) throws DMLRuntimeException {
		boolean skipCache = cache_out == null || cache_c == null || cache_ifog == null;
		
		if( (skipCache && (cache_out != null || cache_c != null || cache_ifog != null)) || 
			(!skipCache && (cache_out == null || cache_c == null || cache_ifog == null))) {
			throw new DMLRuntimeException("Either all cache pointers should be null or all should be not null");
		}
		
		// out_prev = out0
		Pointer out_prev = copy(gCtx, instName, out0, N*M);
		// c_prev = c0
		Pointer c_prev = copy(gCtx, instName, c0, N*M);
		// c = c_prev
		cudaMemcpy(c, c_prev, N*M*sizeOfDataType, cudaMemcpyDeviceToDevice);
		
		Pointer input = gCtx.allocate(instName, N*(D+M)*sizeOfDataType);
		Pointer ifog = gCtx.allocate(instName, N*4*M*sizeOfDataType);
		
		boolean isWSparse = isInSparseFormat(gCtx, W);
		CSRPointer wSparsePointer = null;
		Pointer wDensePointer = null;
		if(isWSparse)
			wSparsePointer = W.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
		else
			wDensePointer = LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, W, instName, D+M, 4*M);
		
		for(int t = 1; t <= T; t++) {
			// X_t = X[,(t-1)*D+1:t*D]  # shape (N, D)
			// input = cbind(X_t, out_prev)  # shape (N, D+M)
			getCudaKernels(gCtx).launchKernel("prepareInputNNLstm",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*(D+M))),
					X, out_prev, input, (t-1), toInt(M), toInt(D), toInt(T*D), toInt(D+M), toInt(N*(D+M)));
			
			// ifog = input %*% W
			CuMatMultParameters param = new CuMatMultParameters(N, D+M,
					D+M, 4*M, false, false);
			if(isWSparse) {
				if(wSparsePointer.nnz == 0) {
					cudaMemset(ifog, 0, N*4*M*sizeOfDataType);
				}
				else {
					LibMatrixCuMatMult.denseSparseMatMult(gCtx.getCusparseHandle(), instName, ifog, input, wSparsePointer, param);
				}
			}
			else
				LibMatrixCuMatMult.denseDenseMatMult(gCtx.getCublasHandle(), instName, ifog, input, wDensePointer, param);
			
			// ifog = ifog + b
			// ifog[,1:3*M] = sigmoid::forward(ifog[,1:3*M])  # i,f,o gates squashed with sigmoid
			// ifog[,3*M+1:4*M] = tanh::forward(ifog[,3*M+1:4*M])  # g gate squashed with tanh
			getCudaKernels(gCtx).launchKernel("squashIFOG",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*4*M)),
				ifog, b, toInt(M), toInt(N*4*M));
			
			
			// c = ifog[,M+1:2*M]*c_prev + ifog[,1:M]*ifog[,3*M+1:4*M]
			// out_t = ifog[,2*M+1:3*M] * tanh::forward(c)
			// if (return_sequences) {
			//   out[,(t-1)*M+1:t*M] = out_t
			// }
			// else {
			//   out = out_t
			// }
			// out_prev = out_t
			// c_prev = c
			// cache_out[t,] = matrix(out_t, rows=1, cols=N*M)
			// cache_c[t,] = matrix(c, rows=1, cols=N*M)
			if(skipCache) {
				getCudaKernels(gCtx).launchKernel("postProcessNNLstmForwardSkipCache",
						ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*M)),
					ifog, c,  out_prev, c_prev, out,
					return_sequences ? 1 : 0, t-1, toInt(T), toInt(M), toInt(N*M));
			}
			else {
				getCudaKernels(gCtx).launchKernel("postProcessNNLstmForward",
						ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*M)),
					ifog, c,  out_prev, c_prev, out, cache_out, cache_c,
					return_sequences ? 1 : 0, t-1, toInt(T), toInt(M), toInt(N*M));
				
				// cache_ifog[t,] = matrix(ifog, rows=1, cols=N*4*M)  # reshape
				cudaMemcpy(cache_ifog.withByteOffset((t-1)*N*4*M*sizeOfDataType), ifog, N*4*M*sizeOfDataType, cudaMemcpyDeviceToDevice);
			}
		}
		
		gCtx.cudaFreeHelper(instName, out_prev, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, c_prev, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, input, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, ifog, gCtx.EAGER_CUDA_FREE);
	}
	
	private static Pointer copy(GPUContext gCtx, String instName, Pointer ptr, long numElems) {
		Pointer ret = gCtx.allocate(instName, numElems*sizeOfDataType);
		cudaMemcpy(ret, ptr, numElems*sizeOfDataType, cudaMemcpyDeviceToDevice);
		return ret;
	}
	
	/**
	 * Computes the forward pass for an LSTM layer with M neurons.
	 * The input data has N sequences of T examples, each with D features.
	 * 
	 * @param ec execution context 
	 * @param gCtx gpu context 
	 * @param instName name of the instruction
	 * @param X input matrix pointer
	 * @param wPointer weight matrix pointer
	 * @param out0 Outputs from previous timestep
	 * @param c0 Initial cell state
	 * @param return_sequences Whether to return `out` at all timesteps, or just for the final timestep.
	 * @param outputName name of the out variable. If `return_sequences` is True, outputs for all timesteps.
	 * @param cyName name of the output cell state. Cell state for final timestep.
	 * @param N minibatch size
	 * @param M hidden size
	 * @param D number of features
	 * @param T sequence length
	 * @param prefixTempCache prefix for temporary cache variable or null if not eligible.
	 * @param scopeVar scope variable for temporary cache
	 * @throws DMLRuntimeException if error
	 */
	public static void cuDNNLstm(ExecutionContext ec, GPUContext gCtx, String instName,
			Pointer X,  Pointer wPointer, Pointer out0, Pointer c0, boolean return_sequences,
			String outputName, String cyName, int N, int M, int D, int T, String prefixTempCache, String scopeVar) throws DMLRuntimeException {
		cuDNNSingleLayerUnidirectionalRNNForward(ec, gCtx, instName, X, out0, c0, wPointer, outputName, cyName, 
				"lstm", return_sequences, N, M, D, T, prefixTempCache, scopeVar);
	}
	
	private static void cuDNNSingleLayerUnidirectionalRNNForward(ExecutionContext ec, GPUContext gCtx, String instName,
			Pointer x, Pointer hx, Pointer cx, Pointer wPointer,  // input
			String outputName, String cyName,  					 // output
			String rnnMode, boolean return_sequences, int N, int M, int D, int T, 
			String prefixTempCache, String scopeVar) throws DMLRuntimeException {
		boolean hasCarry = rnnMode.equalsIgnoreCase("lstm");
		if(LOG.isDebugEnabled()) {
			long memRequired = (N*T*M + 2*N*M + N*T*M)*sizeOfDataType;
			LOG.debug("Memory required for invoking lstmForward is " + memRequired + " bytes + workspace + reserve space + memory for descriptors.");
		}
		
		// Get output pointers
		Pointer cudnnYPointer = gCtx.allocate(instName, N*T*M*sizeOfDataType);
		Pointer hyPointer = return_sequences ? gCtx.allocate(instName, N*M*sizeOfDataType) : getDenseOutputPointer(ec, gCtx, instName, outputName, N, M);
		Pointer cyPointer = hasCarry ? getDenseOutputPointer(ec, gCtx, instName, cyName, N, M) : new Pointer();
		// Pointer wPointer = getDensePointerForCuDNN(gCtx, w, instName, D+M+2, 4*M);
		
		try(LibMatrixCuDNNRnnAlgorithm algo = new LibMatrixCuDNNRnnAlgorithm(ec, gCtx, instName, rnnMode, N, T, M, D, true, wPointer)) {
			Pointer reserveSpace = null;
			MatrixObject reserveSpaceMO = null;
			if (algo.reserveSpaceSizeInBytes != 0) {
				if(ConfigurationManager.allocateNNCache()) {
					Pair<MatrixObject, Pointer> tmp = LibMatrixCuDNN.getTemporaryCacheDensePointer(ec, gCtx, instName, 
							prefixTempCache + "_cudnn_reserveSpace", scopeVar, algo.reserveSpaceSizeInBytes);
					reserveSpaceMO = tmp.getKey();
					reserveSpace = tmp.getValue();
				}
				else {
					reserveSpace = gCtx.allocate(instName, algo.reserveSpaceSizeInBytes);
				}
			}
			else {
				reserveSpace = new Pointer();
			}
			
			JCudnn.cudnnRNNForwardTraining(gCtx.getCudnnHandle(), algo.rnnDesc, T, 
					algo.xDesc, x, 
					algo.hxDesc, hx, 
					algo.cxDesc, cx, 
					algo.wDesc, wPointer, 
					algo.yDesc, cudnnYPointer, 
					algo.hyDesc, hyPointer, 
					algo.cyDesc, cyPointer, 
					algo.workSpace, algo.sizeInBytes, 
					reserveSpace, algo.reserveSpaceSizeInBytes);
			if (algo.reserveSpaceSizeInBytes != 0) {
				if(ConfigurationManager.allocateNNCache()) {
					// Release the temporary cache variable for the reserve space
					ec.releaseTemporaryCacheMatrixForGPUInstruction(reserveSpaceMO);
				}
				else {
					gCtx.cudaFreeHelper(instName, reserveSpace, gCtx.EAGER_CUDA_FREE);
				}
			}
		}
		
		if(return_sequences) {
			gCtx.cudaFreeHelper(instName, hyPointer, gCtx.EAGER_CUDA_FREE);
			Pointer sysmlYPointer = getDenseOutputPointer(ec, gCtx, instName, outputName, N, T*M);
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_output",
					ExecutionConfig.getConfigForSimpleVectorOperations(N*T*M),
					sysmlYPointer, cudnnYPointer, N, T, M, N*T*M);
		}
		gCtx.cudaFreeHelper(instName, cudnnYPointer, gCtx.EAGER_CUDA_FREE);
	}
	
	public static void cuDNNLstmBackward(ExecutionContext ec, GPUContext gCtx, String instName,
			Pointer x, Pointer hx, Pointer cx, Pointer wPointer, String doutName, String dcyName,  // input
			String dxName, String dwName, String dbName, String dhxName, String dcxName,  	// output
			boolean return_sequences, long N, long M, long D, long T, String prefixTempCache, String scopeVar) throws DMLRuntimeException {
		
		if(LOG.isDebugEnabled()) {
			long memRequired = (D+M)*4*M // sysmlWPointer
					+ 2*(D+M+2)*(4*M) // cudnnWPointer and cudnnDwPointer
					+ 3*N*T*D  // cudnnInput, cudnnDx and smlDx
					+ 2*N*T*M // dy and yPointer
					+ (return_sequences ? T*M : M); // dout
			memRequired *= LibMatrixCUDA.sizeOfDataType;
			// Assume the workspace to be proportional to cudnnWPointer
			// memRequired += (D+M+2)*(4*M)*LibMatrixCUDA.sizeOfDataType;
			LOG.debug("Memory required for invoking lstmBackward is " + memRequired + " bytes + workspace + reserve space + memory for descriptors.");
		}
		
		// Transform the input dout and prepare them for cudnnRNNBackwardData
		Pointer dy = gCtx.allocate(instName, N*T*M*sizeOfDataType);
		long size = return_sequences ? N*T*M : N*M;
		LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_backward_gradients",
				ExecutionConfig.getConfigForSimpleVectorOperations(toInt(size)),
				getDenseInputPointer(ec, gCtx, instName, doutName, N, return_sequences ? T*M : M),
				dy, N, T, M, size, return_sequences ? 1 : 0);
		ec.releaseMatrixInputForGPUInstruction(doutName);
				
		// Allocate intermediate pointers computed by forward
		Pointer yPointer = gCtx.allocate(instName, N*T*M*sizeOfDataType);
		
		try(LibMatrixCuDNNRnnAlgorithm algo = new LibMatrixCuDNNRnnAlgorithm(ec, gCtx, instName, "lstm", toInt(N), toInt(T), 
				toInt(M), toInt(D), true, wPointer)) {
			boolean invokeForward = true;
			Pointer reserveSpace = null;
			if (algo.reserveSpaceSizeInBytes != 0) {
				if(ConfigurationManager.allocateNNCache()) {
					if(ec.containsTemporaryCacheMatrix(prefixTempCache + "_cudnn_reserveSpace", scopeVar)) {
						invokeForward = false;
						reserveSpace = LibMatrixCuDNN.getDenseInputPointer(ec, gCtx, instName, prefixTempCache + "_cudnn_reserveSpace");
					}
					else {
						// Only warn when ConfigurationManager.allocateNNCache() is true
						LOG.warn("Invoking cudnnRNNForwardTraining function redundantly in lstm_backward call. "
								+ "Note: This can sometime happen due to copy or move instruction.");
					}
				}
			}
			else {
				invokeForward = false;
				reserveSpace = new Pointer();
			}
			
			if(invokeForward) {
				reserveSpace = new Pointer();
				if (algo.reserveSpaceSizeInBytes != 0) {
					if(LOG.isDebugEnabled()) 
						LOG.debug("Allocating " +  algo.reserveSpaceSizeInBytes + " bytes for lstm reserve space.");
					reserveSpace = gCtx.allocate(instName, algo.reserveSpaceSizeInBytes);
				}
				JCudnn.cudnnRNNForwardTraining(gCtx.getCudnnHandle(), algo.rnnDesc, toInt(T), 
						algo.xDesc, x, 
						algo.hxDesc, hx, 
						algo.cxDesc, cx, 
						algo.wDesc, wPointer, 
						algo.yDesc, yPointer, 
						algo.hyDesc, new Pointer(), 
						algo.cyDesc, new Pointer(), 
						algo.workSpace, algo.sizeInBytes, 
						reserveSpace, algo.reserveSpaceSizeInBytes);
			}
			
			Pointer cudnnDx = gCtx.allocate(instName, N*T*D*LibMatrixCUDA.sizeOfDataType);
			JCudnn.cudnnRNNBackwardData(gCtx.getCudnnHandle(), algo.rnnDesc, toInt(T), 
					algo.yDesc, yPointer,
					// ----------------------
					// Additional inputs:
					algo.dyDesc, dy, 
					algo.dhyDesc, new Pointer(), 
					algo.dcyDesc, getDenseInputPointer(ec, gCtx, instName, dcyName, N, M),
					// ----------------------
					algo.wDesc, wPointer, 
					algo.hxDesc, hx,
					algo.cxDesc, cx,
					// ----------------------
					// Output:
					algo.dxDesc, cudnnDx, 
					algo.dhxDesc, getDenseOutputPointer(ec, gCtx, instName, dhxName, N, M), 
					algo.dcxDesc, getDenseOutputPointer(ec, gCtx, instName, dcxName, N, M),
					// ----------------------
					algo.workSpace, algo.sizeInBytes, 
					reserveSpace, algo.reserveSpaceSizeInBytes);
			gCtx.cudaFreeHelper(instName, dy, gCtx.EAGER_CUDA_FREE);
			ec.releaseMatrixInputForGPUInstruction(dcyName);
			ec.releaseMatrixOutputForGPUInstruction(dhxName);
			ec.releaseMatrixOutputForGPUInstruction(dcxName);
			
			Pointer smlDx = getDenseOutputPointer(ec, gCtx, instName, dxName, N, T*D);
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_dinput",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt(N*T*D)),
					smlDx, cudnnDx, N, D, T*D, N*T*D);
			ec.releaseMatrixOutputForGPUInstruction(dxName);
			gCtx.cudaFreeHelper(instName, cudnnDx, gCtx.EAGER_CUDA_FREE);
			
			// -------------------------------------------------------------------------------------------
			Pointer cudnnDwPointer = gCtx.allocate(instName, (D+M+2)*(4*M)*LibMatrixCUDA.sizeOfDataType);
			JCudnn.cudnnRNNBackwardWeights(gCtx.getCudnnHandle(), algo.rnnDesc, toInt(T), 
					algo.xDesc, x, 
					algo.hxDesc, hx, 
					algo.yDesc, yPointer, 
					algo.workSpace, algo.sizeInBytes, 
					algo.dwDesc, cudnnDwPointer, 
					reserveSpace, algo.reserveSpaceSizeInBytes);
			LibMatrixCUDA.getCudaKernels(gCtx).launchKernel("prepare_lstm_dweight",
					ExecutionConfig.getConfigForSimpleVectorOperations(toInt((D+M+2)*(4*M))),
					getDenseOutputPointer(ec, gCtx, instName, dwName, D+M, 4*M), 
					getDenseOutputPointer(ec, gCtx, instName, dbName, 1, 4*M), cudnnDwPointer, D, M);
			gCtx.cudaFreeHelper(instName, cudnnDwPointer, gCtx.EAGER_CUDA_FREE);
			ec.releaseMatrixOutputForGPUInstruction(dwName);
			ec.releaseMatrixOutputForGPUInstruction(dbName);
			// -------------------------------------------------------------------------------------------
			
			gCtx.cudaFreeHelper(instName, yPointer, gCtx.EAGER_CUDA_FREE);
			if (algo.reserveSpaceSizeInBytes != 0) {
				if(invokeForward) {
					// Free the allocated reserve space
					gCtx.cudaFreeHelper(instName, reserveSpace, gCtx.EAGER_CUDA_FREE);
				}
				else {
					// Release the temporary cache variable for the reserve space
					ec.releaseMatrixInputForGPUInstruction(prefixTempCache + "_cudnn_reserveSpace");
				}
			}
		}
	}
	
	
	
	/**
	 * Performs the forward BatchNormalization layer computation for training
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName name of the instruction
	 * @param image input image
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param bias bias (as per CuDNN) and beta as per original paper: shape [1, C, 1, 1]
	 * @param runningMean running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param runningVar running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param ret (output) normalized input
	 * @param retRunningMean (output) running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param retRunningVar (output) running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @param exponentialAverageFactor factor used in the moving average computation
	 * @param resultSaveMean (output) running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param resultSaveInvVariance (output) running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationForwardTraining(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject scale,  MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar,
			MatrixObject ret, MatrixObject retRunningMean, MatrixObject retRunningVar, 
			double epsilon, double exponentialAverageFactor,
			MatrixObject resultSaveMean, MatrixObject resultSaveInvVariance) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationForwardTraining" + ", GPUContext=" + gCtx);
		}

		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumRows());
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);

		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer retPtr = getDensePointerForCuDNN(gCtx, ret, instName);
		Pointer biasPtr = getDensePointerForCuDNN(gCtx, bias, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer runningMeanPtr = getDensePointerForCuDNN(gCtx, runningMean, instName);
		Pointer runningVarPtr = getDensePointerForCuDNN(gCtx, runningVar, instName);

		// To allow for copy-on-write
		Pointer retRunningMeanPtr = getDensePointerForCuDNN(gCtx, retRunningMean, instName);
		Pointer retRunningVarPtr = getDensePointerForCuDNN(gCtx, retRunningVar, instName);
		cudaMemcpy(retRunningMeanPtr, runningMeanPtr, C * sizeOfDataType, cudaMemcpyDeviceToDevice);
		cudaMemcpy(retRunningVarPtr, runningVarPtr, C * sizeOfDataType, cudaMemcpyDeviceToDevice);

		Pointer resultSaveMeanPtr = getDensePointerForCuDNN(gCtx, resultSaveMean, instName);
		Pointer resultSaveInvVariancePtr = getDensePointerForCuDNN(gCtx, resultSaveInvVariance, instName);
		
		checkStatus(cudnnBatchNormalizationForwardTraining(getCudnnHandle(gCtx), 
				jcuda.jcudnn.cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, biasPtr, exponentialAverageFactor,
				retRunningMeanPtr, retRunningVarPtr, epsilon, resultSaveMeanPtr, resultSaveInvVariancePtr));
	}
	
	/**
	 * Performs the forward BatchNormalization layer computation for inference
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName name of the instruction
	 * @param image input image
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param bias bias (as per CuDNN) and beta as per original paper: shape [1, C, 1, 1]
	 * @param runningMean running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param runningVar running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @param ret normalized input
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationForwardInference(GPUContext gCtx, String instName, MatrixObject image,
			MatrixObject scale, MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar,
			MatrixObject ret, double epsilon) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationForwardInference" + ", GPUContext=" + gCtx);
		}

		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumRows());
		long CHW = image.getNumColumns();
		validateBatchNormalizationDimensions(scale, bias, runningMean, runningVar, C);

		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image},  new MatrixObject[] {ret});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer retPtr = getDensePointerForCuDNN(gCtx, ret, instName);
		Pointer biasPtr = getDensePointerForCuDNN(gCtx, bias, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer runningMeanPtr = getDensePointerForCuDNN(gCtx, runningMean, instName);
		Pointer runningVarPtr = getDensePointerForCuDNN(gCtx, runningVar, instName);

		checkStatus(cudnnBatchNormalizationForwardInference(getCudnnHandle(gCtx), 
				jcuda.jcudnn.cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL, one(), zero(),
				nCHWDescriptor, imagePtr, nCHWDescriptor, retPtr,
				scaleTensorDesc, scalePtr, biasPtr,
				runningMeanPtr, runningVarPtr, epsilon));
	}
	
	private static void validateDimensions(MatrixObject mo, long expectedRows, long expectedCols) {
		if(mo.getNumRows() != expectedRows || mo.getNumColumns() != expectedCols) {
			throw new DMLRuntimeException("Incorrect dimensions for the input matrix object. Expected [" + expectedRows + ", " +  expectedCols+ "], but found "
					+ "[" + mo.getNumRows() + ", " + mo.getNumColumns() + "].");
		}
	}
	
	/**
	 * This method computes the backpropagation errors for image of batch normalization layer
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName name of the instruction
	 * @param image input image
	 * @param dout input errors of shape C, H, W
	 * @param scale scale (as per CuDNN) and gamma as per original paper: shape [1, C, 1, 1]
	 * @param dX (output) backpropagation errors for previous layer
	 * @param epsilon epsilon value used in the batch normalization formula
	 * @param resultSaveMean (input) running mean accumulated during training phase: shape [1, C, 1, 1]
	 * @param resultSaveInvVariance (input) running variance accumulated during training phase: shape [1, C, 1, 1]
	 * @throws DMLRuntimeException if error occurs
	 */
	public static void batchNormalizationBackwardDX(GPUContext gCtx, String instName, MatrixObject image, MatrixObject dout,
			MatrixObject scale, MatrixObject dX, 
			// MatrixObject dScale, MatrixObject dBias,
			double epsilon, MatrixObject resultSaveMean, MatrixObject resultSaveInvVariance) throws DMLRuntimeException {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : batchNormalizationBackward" + ", GPUContext=" + gCtx);
		}
		
		int N = toInt(image.getNumRows());
		int C = toInt(scale.getNumRows());
		long CHW = image.getNumColumns();
		
		validateDimensions(scale, C, 1);
		validateDimensions(dX, N, CHW);
		validateDimensions(dout, N, CHW);
		validateDimensions(resultSaveMean, C, 1);
		validateDimensions(resultSaveInvVariance, C, 1);
		
		// Allocate descriptors
		cudnnTensorDescriptor nCHWDescriptor = allocateNCHWDescriptors(gCtx, N, C, CHW,
				new MatrixObject[] {image, dout},  new MatrixObject[] {dX});
		cudnnTensorDescriptor scaleTensorDesc = allocateTensorDescriptor(1, C, 1, 1);

		// Get underlying dense pointer
		Pointer imagePtr = getDensePointerForCuDNN(gCtx, image, instName);
		Pointer doutPtr = getDensePointerForCuDNN(gCtx, dout, instName);
		Pointer scalePtr = getDensePointerForCuDNN(gCtx, scale, instName);
		Pointer dXPtr = getDensePointerForCuDNN(gCtx, dX, instName);
		Pointer dScalePtr = gCtx.allocate(instName, C*LibMatrixCUDA.sizeOfDataType); // getDensePointerForCuDNN(gCtx, dScale, instName);
		Pointer dBiasPtr = gCtx.allocate(instName, C*LibMatrixCUDA.sizeOfDataType); //getDensePointerForCuDNN(gCtx, dBias, instName);		
		Pointer resultSaveMeanPtr = getDensePointerForCuDNN(gCtx, resultSaveMean, instName);
		Pointer resultSaveInvVariancePtr = getDensePointerForCuDNN(gCtx, resultSaveInvVariance, instName);

		cudnnBatchNormalizationBackward(getCudnnHandle(gCtx), 
				jcuda.jcudnn.cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL,  one(), zero(), one(), zero(),
				nCHWDescriptor,  imagePtr, nCHWDescriptor, doutPtr, nCHWDescriptor, dXPtr,
				scaleTensorDesc, scalePtr, dScalePtr, dBiasPtr, epsilon, resultSaveMeanPtr, resultSaveInvVariancePtr);
		gCtx.cudaFreeHelper(instName, dScalePtr, gCtx.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, dBiasPtr, gCtx.EAGER_CUDA_FREE);
	}
	
	private static void validateBatchNormalizationDimensions(MatrixObject scale, MatrixObject bias, MatrixObject runningMean, MatrixObject runningVar, int C) throws DMLRuntimeException {
		if(scale.getNumRows() != C || scale.getNumColumns() != 1) {
			throw new DMLRuntimeException("Incorrect dimensions for scale. Expected a column vector of size " + C + ", but found [" + scale.getNumRows() + ", " + scale.getNumColumns() + "]");
		}
		if(bias.getNumRows() != C || bias.getNumColumns() != 1) {
			throw new DMLRuntimeException("Incorrect dimensions for bias. Expected a column vector of size " + C + ", but found [" + bias.getNumRows() + ", " + bias.getNumColumns() + "]");
		}
		if(runningMean.getNumRows() != C || runningMean.getNumColumns() != 1) {
			throw new DMLRuntimeException("Incorrect dimensions for running mean. Expected a column vector of size " + C + ", but found [" + runningMean.getNumRows() + ", " + runningMean.getNumColumns() + "]");
		}
		if(runningVar.getNumRows() != C || runningVar.getNumColumns() != 1) {
			throw new DMLRuntimeException("Incorrect dimensions for running variance. Expected a column vector of size " + C + ", but found [" + runningVar.getNumRows() + ", " + runningVar.getNumColumns() + "]");
		}
	}
	
	/**
	 * Convenient utility for batch normalization that returns a NCHW descriptor
	 * @param gCtx a valid {@link GPUContext}
	 * @param N number of images
	 * @param C number of channels
	 * @param CHW channels*height*width
	 * @param input input matrix objects
	 * @param output output matrix objects
	 * @return one of the NCHW descriptor
	 * @throws DMLRuntimeException if error occurs
	 */
	private static cudnnTensorDescriptor allocateNCHWDescriptors(GPUContext gCtx, int N, int C, long CHW, MatrixObject [] input, MatrixObject [] output) throws DMLRuntimeException {
		cudnnTensorDescriptor ret  = null; // Return any one
		if(CHW > ((long)Integer.MAX_VALUE)*C) {
			throw new DMLRuntimeException("image size (height*width) should be less than " + Integer.MAX_VALUE);
		}
		int H = -1; int W = -1;
		int HW = (int) (CHW / C);
		H = HW; W = 1; // If not known
		double potentialH = Math.sqrt(HW);
		if(potentialH == ((int) potentialH)) {
			H = (int) potentialH;
			W = H;
		}
		// We are not sure about H and W, hence don't allocate them.
		ret = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(ret);
		cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_TYPE, N, C, H, W);
		return ret;
	}
	
	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param image input matrix object
	 * @param instName name of the instruction
	 * @return jcuda pointer
	 */
	public static Pointer getDensePointerForCuDNN(GPUContext gCtx, MatrixObject image, String instName) {
		long numElems = image.getNumRows()*image.getNumColumns();
		if(numElems > maxNumElementsOfCuDNNTensor) {
			throw new DMLRuntimeException("CuDNN restriction: the size of input tensor cannot have greater than 2 giga-elements, but has " + numElems + " (i.e. [" + image.getNumRows() + " X " + image.getNumColumns() + "]). Hint: try reducing the mini-batch size.");
		}
		return getDensePointer(gCtx, image, instName);
	}
	
	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param image input matrix object
	 * @param instName name of the instruction
	 * @param numRows expected number of rows
	 * @param numCols expected number of columns 
	 * @return jcuda pointer
	 * @throws DMLRuntimeException if error occurs while sparse to dense conversion
	 */
	public static Pointer getDensePointerForCuDNN(GPUContext gCtx, MatrixObject image, String instName, long numRows, long numCols) throws DMLRuntimeException {
		long numElems = image.getNumRows()*image.getNumColumns();
		if(image.getNumRows() != numRows || image.getNumColumns() != numCols) {
			throw new DMLRuntimeException("Expected input of size:[" +  numRows + ", " + numCols + "], but found [" + image.getNumRows() + ", " + image.getNumColumns() + "]."); 
		}
		else if(numElems > maxNumElementsOfCuDNNTensor) {
			throw new DMLRuntimeException("CuDNN restriction: the size of input tensor cannot have greater than 2 giga-elements, but has " + numElems + " (i.e. [" + image.getNumRows() + " X " + image.getNumColumns() + "]). Hint: try reducing the mini-batch size.");
		}
		Pointer ptr = getDensePointer(gCtx, image, instName);
		long sizeOfPtr = gCtx.getMemoryManager().getSizeAllocatedGPUPointer(ptr);
		if(sizeOfPtr != numElems*sizeOfDataType) {
			throw new DMLRuntimeException("Incorrect pointer: expected size:" +  (numElems*sizeOfDataType) + ", but found " + sizeOfPtr);
		}
		return ptr;
	}

	/**
	 * Convenience method for checking the status of CuDNN kernel.
	 *
	 * @param status status returned by CuDNN
	 */
	protected static void checkStatus(int status) {
		if(status != cudnnStatus.CUDNN_STATUS_SUCCESS)
			throw new DMLRuntimeException("Error status returned by CuDNN:" + jcuda.jcudnn.cudnnStatus.stringFor(status));
	}
}

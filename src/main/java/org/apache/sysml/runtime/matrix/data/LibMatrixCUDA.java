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

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardData;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionBackwardFilter;
import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardDataWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolution2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetFilter4dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetPooling2dDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetTensor4dDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDcsrmv;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.And;
import org.apache.sysml.runtime.functionobjects.Divide;
import org.apache.sysml.runtime.functionobjects.Equals;
import org.apache.sysml.runtime.functionobjects.GreaterThan;
import org.apache.sysml.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysml.runtime.functionobjects.LessThan;
import org.apache.sysml.runtime.functionobjects.LessThanEquals;
import org.apache.sysml.runtime.functionobjects.Minus;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Multiply2;
import org.apache.sysml.runtime.functionobjects.NotEquals;
import org.apache.sysml.runtime.functionobjects.Or;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.Power;
import org.apache.sysml.runtime.functionobjects.Power2;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaKernels;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject;
import org.apache.sysml.runtime.instructions.gpu.context.JCudaObject.CSRPointer;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysml.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.utils.Statistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnConvolutionFwdPreference;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;

//FIXME move could to respective instructions, this is not a block library
public class LibMatrixCUDA {
	
	public static cudnnHandle cudnnHandle;
	public static cublasHandle cublasHandle;
	public static cusparseHandle cusparseHandle;
	public static JCudaKernels kernels; // Used to launch custom kernels

    private static final Log LOG = LogFactory.getLog(LibMatrixCUDA.class.getName());

	private static int CONVOLUTION_PREFERENCE = cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
	
	public static void conv2d(MatrixObject image, MatrixObject filter, MatrixObject outputBlock, int N, int C, int H, int W,
			int K, int R, int S, int pad_h, int pad_w, int stride_h, int stride_w, int P, int Q)
			throws DMLRuntimeException {
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
			srcTensorDesc = allocateTensorDescriptor(N, C, H, W);
			dstTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			filterDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			// (Pointer) gpuCtx.prepare(image, true, true);
			// (Pointer) gpuCtx.prepare(filter, true, true);
			
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer filterPointer = ((JCudaObject)filter.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dstPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			
			// Select the best algorithm depending on the data and supported CUDA
			
			int algo = -1; 
			workSpace = new Pointer();
			
			if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE) {
				algo = jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			}
			else if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST) {
				int [] algos = {
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
	            		jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	            };
				// TODO: Look into FFt, Winograd, etc
				// Also ensure that GPU has enough memory to allocate memory
				long sizeInBytesArray[] = { 0 };
	            algo = jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
	            		CONVOLUTION_PREFERENCE, sizeInBytesArray[0], algos);
	            cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo, sizeInBytesArray);
	            if(sizeInBytesArray[0] != 0)
	            	jcuda.runtime.JCuda.cudaMalloc(workSpace, sizeInBytesArray[0]);
	            sizeInBytes = sizeInBytesArray[0];
			}
			else if(CONVOLUTION_PREFERENCE == cudnnConvolutionFwdPreference.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT) {
				throw new DMLRuntimeException("CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT is not implemented");
			}
			else {
				throw new DMLRuntimeException("Unsupported preference criteria for convolution");
			}
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			int status = cudnnConvolutionForward(cudnnHandle, alpha, 
					srcTensorDesc, imagePointer, 
					filterDesc, filterPointer,
					convDesc, algo, workSpace, sizeInBytes, beta,
					dstTensorDesc, dstPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
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
	
	private static cudnnConvolutionDescriptor allocateConvolutionDescriptor(int padding [], int strides []) {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		cudnnSetConvolution2dDescriptor(convDesc, padding[0], padding[1], strides[0], strides[1], 1, 1, CUDNN_CROSS_CORRELATION);		
		return convDesc;
	}
	
	public static  Pointer pointerTo(double value) {
        return Pointer.to(new double[] { value });
    }
	
	private static  cudnnTensorDescriptor allocateTensorDescriptor(int N, int C, int H, int W) {
		cudnnTensorDescriptor ret = new cudnnTensorDescriptor();
		cudnnCreateTensorDescriptor(ret);
		cudnnSetTensor4dDescriptor(ret, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W);
		return ret;
	}
	
	private static cudnnFilterDescriptor allocateFilterDescriptor(int K, int C, int R, int S) {
		cudnnFilterDescriptor filterDesc = new cudnnFilterDescriptor();
		cudnnCreateFilterDescriptor(filterDesc);
		cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, K, C, R, S);
		return filterDesc;
	}
	
	/**
	 * allocates pooling descriptor, used in poolingForward and poolingBackward
	 * @param R			pooling window height
	 * @param S			pooling window width
	 * @param pad_h		vertical padding
	 * @param pad_w		horizontal padding
	 * @param stride_h	pooling vertical stride
	 * @param stride_w	pooling horizontal stride
	 * @return
	 */
	private static cudnnPoolingDescriptor allocatePoolingDescriptor(int R, int S, int pad_h, int pad_w, int stride_h, int stride_w) {
		cudnnPoolingDescriptor poolingDesc = new cudnnPoolingDescriptor();
		cudnnCreatePoolingDescriptor(poolingDesc);
		cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, R, S, pad_h, pad_w, stride_h, stride_w);
		return poolingDesc;
	}
	
	public static void conv2d_backward_filter(MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
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
			xTensorDesc = allocateTensorDescriptor(N, C, H, W);
			doutTensorDesc = allocateTensorDescriptor(N, K, P, Q);
			dwDesc = allocateFilterDescriptor(K, C, R, S);
			
			// Allocate data
			Pointer imagePointer = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer doutPointer = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dwPointer = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
					xTensorDesc, doutTensorDesc, convDesc, dwDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardFilter(cudnnHandle, alpha, xTensorDesc, imagePointer, 
					doutTensorDesc, doutPointer, convDesc, algo, workSpace, sizeInBytes, beta, dwDesc, dwPointer);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardFilter: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
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

	/**
	 * Performs tsmm, A %*% A' or A' %*% A, on GPU by exploiting cublasDsyrk(...) 
	 * @param left	input matrix, as in a tsmm expression like A %*% A' or A' %*% A, we just need to check whether the left one is transposed or not, I named it 'left'
	 * @param output
	 * @param isLeftTransposed
	 * @throws DMLRuntimeException
	 */
	public static void matmultTSMM(ExecutionContext ec, MatrixObject left, String outputName,
            boolean isLeftTransposed) throws DMLRuntimeException {
	    if(isInSparseFormat(left)) {
	    	// For sparse TSMM, invoke matmult (TODO: possible performance improvement)
	    	matmult(ec, left, left, outputName, isLeftTransposed, !isLeftTransposed);
	    	return;
	    }
	
	    // For dense TSMM, exploit cublasDsyrk(...) and call custom kernel to flip the matrix
	    MatrixObject output = ec.getMatrixObject(outputName);
	    ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
	    
	    // Since CuBLAS expects inputs in column-major format,
	    // reverse the order of matrix-multiplication and take care of dimension mismatch.      
	    int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
	    // Note: the dimensions are swapped
	    int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows());
	    int k = (int) (isLeftTransposed ? left.getNumRows() : left.getNumColumns());
	
	    if(m == -1)
	            throw new DMLRuntimeException("Incorrect dimensions");
	
	    double[] alpha = {1.0d};
	    double[] beta = {0.0d};
	
	    int lda = (int) (isLeftTransposed ? m : k);
	    int ldc = m;
	
	    if(!left.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Input is not allocated:" + left.getGPUObject().isAllocated);
	    if(!output.getGPUObject().isAllocated)
	            throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject().isAllocated);
	
	    Pointer A = ((JCudaObject)left.getGPUObject()).jcudaDenseMatrixPtr;
	    Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;
	    
	    JCublas2.cublasDsyrk(cublasHandle, cublasFillMode.CUBLAS_FILL_MODE_LOWER,transa, m, k, Pointer.to(alpha), A, lda, Pointer.to(beta), C, ldc);
	    copyUpperToLowerTriangle(output);
	}
	
	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 * 
	 * @param ret
	 * @throws DMLRuntimeException
	 */
	private static void copyUpperToLowerTriangle(MatrixObject ret) throws DMLRuntimeException {
		if(isInSparseFormat(ret)) {
            throw new DMLRuntimeException("Sparse GPU copyUpperToLowerTriangle is not implemented");
		}
		if(ret.getNumRows() != ret.getNumColumns()) {
			throw new DMLRuntimeException("Only square matrix kernel is implemented for copyUpperToLowerTriangle");
		}
		int dim = (int) ret.getNumRows();
		kernels.launchKernel("copyUpperToLowerTriangleDense",
				ExecutionConfig.getConfigForSimpleMatrixOperations(dim, dim), 
				((JCudaObject)ret.getGPUObject()).jcudaDenseMatrixPtr, dim, dim*dim);
	}
	
	/**
	 * Matrix multiply on GPU
	 * Examines sparsity and shapes and routes call to appropriate method
	 * from cuBLAS or cuSparse
	 * C = op(A) x op(B)
	 * @param ec					Current {@link ExecutionContext} instance
	 * @param left1					Matrix A
	 * @param right1				Matrix B
	 * @param outputName			Name of the output matrix C (in code generated after LOP layer)
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, tranposed or not
	 * @return	output of matrix multiply
	 * @throws DMLRuntimeException
	 */
	public static MatrixObject matmult(ExecutionContext ec, MatrixObject left1, MatrixObject right1, String outputName,
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		
		if(!left1.getGPUObject().isAllocated() || !right1.getGPUObject().isAllocated())
			throw new DMLRuntimeException("One of input is not allocated:" + left1.getGPUObject().isAllocated() + " " + right1.getGPUObject().isAllocated());
		
		boolean bothDense = !left1.getGPUObject().isInSparseFormat() && !right1.getGPUObject().isInSparseFormat();
		boolean bothSparse = left1.getGPUObject().isInSparseFormat() && right1.getGPUObject().isInSparseFormat();
		
		MatrixObject output = ec.getMatrixObject(outputName);

		if (bothDense) {		// Dense C = Dense A * Dense B
			// For both dense, do cuBLAS
			ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
			denseDenseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else if (bothSparse){	// Sparse C = Sparse A * Sparse B
			ec.allocateGPUMatrixObject(outputName);
			bothSparseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		else {	// Either of A or B is sparse, Sparse C = Sparse/Dense A * Dense/Sparse B
				// Convert the dense to sparse and use the cusparseDcsrgemm routine
			ec.allocateGPUMatrixObject(outputName);
			eitherSparseMatmult(output, left1, right1, isLeftTransposed1, isRightTransposed1);
		}
		
		return output;
	}
	
	/**
	 * One of the matrices is sparse, the other dense
	 * C = op(A) x op(B)
	 * @param output				allocated output object for C on host to which GPU output will be attached
	 * @param left					Matrix A on host
	 * @param right					Matrix B on host
	 * @param isLeftTransposed		op for A, tranposed or not
	 * @param isRightTransposed		op for B, transposed or not
	 * @throws DMLRuntimeException
	 */
	protected static void eitherSparseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException {
		
		int transA = isLeftTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		int transB = isRightTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		
		int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows()) ;
		int n = (int) (isRightTransposed ? right.getNumRows() : right.getNumColumns());
		int k = (int) (isLeftTransposed ? left.getNumRows() :  left.getNumColumns());
		int k1 = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		
		if (left.getGPUObject().isInSparseFormat()) {	
			// Left sparse, right dense
			sparseDenseMatmult(output, left, right, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		} else {
			// Left dense, right sparse
			denseSparseMatmult(output, right, left, isLeftTransposed, isRightTransposed, transA, transB, m, n, k);
		}
	}
	
	/**
	 * C = op(A) * op(B) where A is dense and B is sparse
	 * If B is ultrasparse, A is converted to a sparse matrix and {@link #sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise B is converted to a dense matrix and {@link #denseDenseMatmult(MatrixObject, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param output
	 * @param right
	 * @param left
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param transA
	 * @param transB
	 * @param m
	 * @param n
	 * @param k
	 * @throws DMLRuntimeException
	 */
	protected static void denseSparseMatmult(MatrixObject output, MatrixObject right, MatrixObject left,
			boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
			throws DMLRuntimeException {
		// right sparse, left dense
		CSRPointer B = ((JCudaObject)right.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer ADense = ((JCudaObject)left.getGPUObject()).jcudaDenseMatrixPtr;
		if (B.isUltraSparse(k, n)){
			LOG.debug(" GPU Dense-Sparse Matrix Multiplication (Converted to Sparse-Sparse)");
			// Convert left to CSR and do cuSparse matmul
			long t0 = System.nanoTime();
			CSRPointer A = JCudaObject.denseToSparseUtil(cusparseHandle, (int)left.getNumRows(), (int)right.getNumColumns(), ADense);
			Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
			Statistics.cudaConversionCount.addAndGet(1);
			sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
			A.deallocate();
		} else {
			LOG.debug(" GPU Dense-Sparse Matrix Multiplication (Converted to Dense-Dense)");
			// Convert right to dense and do a cuBlas matmul
			// BDenseTransposed is a column major matrix
			// Note the arguments to denseDenseMatmult to accommodate for this.
			Pointer BDenseTransposed = B.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, (int)right.getNumRows(), (int)right.getNumColumns());
			output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
			Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
			denseDenseMatmult(C, 
					(int) left.getNumRows(), (int) left.getNumColumns(),
					(int) right.getNumColumns(), (int) right.getNumRows(), 
					isLeftTransposed, !isRightTransposed,
					ADense, BDenseTransposed);
			cudaFree(BDenseTransposed);
		}
	}

	/**
	 * * C = op(A) * op(B) where A is sparse and B is dense
	 * If A is ultrasparse, B is converted to a sparse matrix and {@link #sparseSparseMatmult(MatrixObject, int, int, int, int, int, CSRPointer, CSRPointer)} is invoked
	 * otherwise A is converted to a dense matrix and {@link #denseDenseMatmult(MatrixObject, int, int, int, int, boolean, boolean, Pointer, Pointer)} is invoked.
	 * @param output
	 * @param left
	 * @param right
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param transA
	 * @param transB
	 * @param m
	 * @param n
	 * @param k
	 * @throws DMLRuntimeException
	 */
	protected static void sparseDenseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed, int transA, int transB, int m, int n, int k)
			throws DMLRuntimeException {
		CSRPointer A = ((JCudaObject)left.getGPUObject()).jcudaSparseMatrixPtr;
		Pointer BDense = ((JCudaObject)right.getGPUObject()).jcudaDenseMatrixPtr;
		
		if (n == 1){	
			// Sparse Matrix - Dense Vector multiply
			LOG.debug(" GPU Sparse Matrix - Dense Vector Mutliply");
			sparseMatrixDenseVectorMult(output, A, BDense, transA, (int)left.getNumRows(), (int)left.getNumColumns());
			
		} else {
			// Sparse Matrix Dense Matrix multiply
			if (A.isUltraSparse(m, k)){	
				LOG.debug(" GPU Sparse-Dense Matrix Multiplication (Converted to Sparse-Sparse)");
				// Convert right to CSR and do cuSparse matmul
				long t0 = System.nanoTime();
				CSRPointer B = JCudaObject.denseToSparseUtil(cusparseHandle, (int)right.getNumRows(), (int)right.getNumColumns(), BDense);
				Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
				Statistics.cudaConversionCount.addAndGet(1);
				sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
				B.deallocate();
			} else {					
				LOG.debug(" GPU Sparse-Dense Matrix Multiplication (Converted to Dense-Dense)");
				// Convert left to dense and do a cuBlas matmul
				// ADenseTransposed is a column major matrix
				// Note the arguments to denseDenseMatmult to accommodate for this.
				Pointer ADenseTransposed = A.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, (int)left.getNumRows(), (int)left.getNumColumns());
				output.getGPUObject().acquireDeviceModifyDense();	// To allocate the dense matrix
				Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
				denseDenseMatmult(C, 
						(int) left.getNumColumns(), (int) left.getNumRows(),
						(int) right.getNumRows(), (int) right.getNumColumns(), 
						!isLeftTransposed, isRightTransposed,
						ADenseTransposed, BDense);
				cudaFree(ADenseTransposed);
			}
		}
	}

	/**
	 * C = op(A) x B
	 * A is a sparse matrix, B is a dense vector
	 * @param output	allocated output on the host, to which the GPU output C will be attached
	 * @param A			sparse matrix A on the GPU
	 * @param B_dense	dense matrix/vector B on the GPU
	 * @param transA	op for A, tranposed or not
	 * @param m			number of rows in A (not op(A))
	 * @param k			number of cols in A or number of rows in B (not op(A) or op(B))
	 * @throws DMLRuntimeException
	 */
	protected static void sparseMatrixDenseVectorMult(MatrixObject output, CSRPointer A, Pointer B_dense, int transA,
			int m, int k) throws DMLRuntimeException {
		long size = m * Sizeof.DOUBLE;
		if (transA == CUSPARSE_OPERATION_TRANSPOSE){
			size = k * Sizeof.DOUBLE;
		}
		Pointer C_dense = JCudaObject.allocate((int)size);
		double[] alpha = { 1 };
		double[] beta = { 0 };
		cusparseDcsrmv(cusparseHandle, transA, m, k, (int)A.nnz, Pointer.to(alpha), A.descr, A.val, A.rowPtr, A.colInd, B_dense, Pointer.to(beta), C_dense);
		cudaDeviceSynchronize(); 	// Since cusparseDcsrmv is asynchronously executed
		((JCudaObject)(output.getGPUObject())).setDenseMatrixCudaPointer(C_dense);
		output.getGPUObject().setDeviceModify(size);
	}

	/**
	 * Sparse C = Sparse op(A) * Sparse op(B)
	 * Reroutes call to sparse matrix-vector mult if needed
	 * @param output
	 * @param left
	 * @param right
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @throws DMLRuntimeException
	 */
	protected static void bothSparseMatmult(MatrixObject output, MatrixObject left, MatrixObject right,
			boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException {
		
		int transA = isLeftTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		int transB = isRightTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
		
		int m = (int) (isLeftTransposed ? left.getNumColumns() : left.getNumRows()) ;
		int n = (int) (isRightTransposed ? right.getNumRows() : right.getNumColumns());
		int k = (int) (isLeftTransposed ? left.getNumRows() :  left.getNumColumns());
		int k1 = (int) (isRightTransposed ? right.getNumColumns() : right.getNumRows());
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
			
		CSRPointer A = ((JCudaObject)left.getGPUObject()).jcudaSparseMatrixPtr;
		CSRPointer B = ((JCudaObject)right.getGPUObject()).jcudaSparseMatrixPtr;
		
		// TODO if (m == 1) {	// Vector-matrix multiplication
		
		if (!isRightTransposed && right.getNumColumns() == 1){ 	// Matrix-Vector multiplication
			sparseMatrixVectorMult(output, transA, (int)left.getNumRows(), (int)left.getNumColumns(), (int)right.getNumRows(), A, B);
		} else {												// Matrix-Matrix multiplication
			sparseSparseMatmult(output, transA, transB, m, n, k, A, B);
		}
	}

	/**
	 * Does a sparse matrix-vector multiply.
	 * C = op(A) x B, A is a sparse matrix, B is a sparse vector with numCols = 1.
	 * @param output	allocated output object C to which the GPU output matrix will be attached
	 * @param transA	if A is to be transposed or not (the op in op(A))
	 * @param m			number of rows in A (not op(A))
	 * @param n			number of cols in A (not op(A))
	 * @param k			number of rows in B, (cols in B is assumed to be 1)		
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse vector on GPU
	 * @throws DMLRuntimeException
	 */
	protected static void sparseMatrixVectorMult(MatrixObject output, int transA, int m, int n, int k,
			CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.debug(" GPU Sparse Matrix Sparse Vector Multiply (Converted to Sparse Matrix Dense Vector Multiply)");
		Pointer BDenseVector = B.toColumnMajorDenseMatrix(cusparseHandle, cublasHandle, k, 1);
		sparseMatrixDenseVectorMult(output, A, BDenseVector, transA, m, k);
	}

	/**
	 * Does a sparse-sparse Matrix multiply
	 * C = op(A) x op(B), A, B are sparse matrices
	 * @param output	allocated output object on host to which the GPU output matrix will be attached
	 * @param transA	op for A - to be transposed or not
	 * @param transB	op for B
	 * @param m			number of rows in op(A)
	 * @param n			number of cols in op(B)
	 * @param k			number of cols in op(A) or rows in op(B)
	 * @param A			left sparse matrix on GPU
	 * @param B			right sparse matrix on GPU
	 * @throws DMLRuntimeException
	 */
	protected static void sparseSparseMatmult(MatrixObject output, int transA, int transB, int m, int n, int k,
			CSRPointer A, CSRPointer B) throws DMLRuntimeException {
		LOG.debug(" GPU Sparse-Sparse Matrix Multiply ");

		CSRPointer C = CSRPointer.allocateForMatrixMultiply(cusparseHandle, A, transA, B, transB, m, n, k);
		((JCudaObject)output.getGPUObject()).setSparseMatrixCudaPointer(C);
		long sizeOfC = CSRPointer.estimateSize(C.nnz, output.getNumRows());
		output.getGPUObject().setDeviceModify(sizeOfC);
		
		cusparseDcsrgemm(cusparseHandle, transA, transB, m, n, k,
				A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd,
				B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd,
				C.descr, C.val, C.rowPtr, C.colInd);
	}

	/**
	 * Dense dense matrix multiply
	 * C = op(A) * op(B), A and B are dense matrices
	 * @param output				output object C on host with GPU data allocated				
	 * @param left1					left matrix A on host (in row-major order)
	 * @param right1				right matrix B on host (in row-major order)
	 * @param isLeftTransposed1 	op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected static void denseDenseMatmult(MatrixObject output, MatrixObject left1, MatrixObject right1,
			boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
		
		Pointer leftPtr = ((JCudaObject)left1.getGPUObject()).jcudaDenseMatrixPtr;
		Pointer rightPtr = ((JCudaObject)right1.getGPUObject()).jcudaDenseMatrixPtr;
		
		int leftRows = (int) left1.getNumRows();
		int leftCols = (int) left1.getNumColumns();
		int rightRows = (int) right1.getNumRows();
		int rightCols = (int) right1.getNumColumns();
		Pointer C = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr;		
		denseDenseMatmult(C, leftRows, leftCols, rightRows, rightCols, isLeftTransposed1, isRightTransposed1,
				leftPtr, rightPtr);
	}

	/**
	 * Dense-dense matrix multiply
	 * C = op(A) * op(B), A and B are dense matrices
	 * On the host, the matrices are in row-major format; cuBLAS expects them in column-major format.
	 * What we have as input is t(A) and t(B), t(X) = transpose of X.
	 * We do t(B) %*% t(A) to get t(C); 
	 * If we were to calculate t(t(C), we would get the resultant matrix C, but this would be in column-major format.
	 * What we really want is t(C). This we already have as the result of t(B) %*% t(A).
	 * @param output			output allocated on GPU in column major format
	 * @param leftRows1			number of rows in A
	 * @param leftCols1			number of cols in A
	 * @param rightRows1		number of rows in B
	 * @param rightCols1		number of cols in B
	 * @param isLeftTransposed1		op for A, transposed or not
	 * @param isRightTransposed1	op for B, transposed or not
	 * @param leftPtr			A allocated on the GPU in row-major format
	 * @param rightPtr			B allocated on the GPU in row-major format
	 * @throws DMLRuntimeException
	 */
	public static void denseDenseMatmult(Pointer output, int leftRows1, int leftCols1, int rightRows1,
			int rightCols1, boolean isLeftTransposed1, boolean isRightTransposed1, Pointer leftPtr, Pointer rightPtr)
			throws DMLRuntimeException {
		
		Pointer A = rightPtr;
		Pointer B = leftPtr;
		
		int leftRows = rightCols1;
		int leftCols = rightRows1;
		int rightRows = leftCols1;
		int rightCols = leftRows1;
		
		boolean isLeftTransposed = isRightTransposed1; 
		boolean isRightTransposed = isLeftTransposed1; 
		
		// Note: the dimensions are swapped
		int m = (int) (isLeftTransposed ? leftCols : leftRows) ;
		int n = (int) (isRightTransposed ? rightRows : rightCols);
		int k = (int) (isLeftTransposed ?  leftRows : leftCols);
		int k1 = (int) (isRightTransposed ?  rightCols : rightRows);
		if(k != k1) 
			throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1);
		
		if(m == -1 || n == -1 || k == -1)
			throw new DMLRuntimeException("Incorrect dimensions");
		
		double[] one = { 1 };
		double[] zero = { 0 };
		
		//int lda = leftRows;
		//int ldb = leftCols;
        int lda = isLeftTransposed ?  k : m;
        int ldb = isRightTransposed ? n : k;
		int ldc = m;
		
		int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
		int transb = isRightTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;

		Pointer C = output;
		if (m == 1 && n == 1){ 
			// Vector product
			LOG.debug(" GPU Dense-dense Vector Product");
			double[] result = {0};
			JCublas2.cublasDdot(cublasHandle, k, A, 1, B, 1, Pointer.to(result));
			// By default in CuBlas V2, cublas pointer mode is set to CUBLAS_POINTER_MODE_HOST.
			// This means that scalar values passed are on host (as opposed to on device).
			// The result is copied from the host back to the device so that the rest of 
			// infrastructure can treat it uniformly.
			cudaMemcpy(C, Pointer.to(result), 1 * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		} else if (m == 1) {
			// Vector-matrix multiply
			LOG.debug(" GPU Dense Vector-Matrix Multiply");
			transb = isRightTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
			JCublas2.cublasDgemv(cublasHandle, transb, rightRows, rightCols, Pointer.to(one), B, ldb, A, 1, Pointer.to(zero), C, 1);
		} else if (n == 1){
			// Matrix-vector multiply
			LOG.debug(" GPU Dense Matrix-Vector Multiply");
			JCublas2.cublasDgemv(cublasHandle, transa, leftRows, leftCols, Pointer.to(one), A, lda, B, 1, Pointer.to(zero), C, 1);
		} else {
			LOG.debug(" GPU Dense-Dense Matrix Multiply ");
			JCublas2.cublasDgemm(cublasHandle, transa, transb, m, n, k, Pointer.to(one), A, lda, B, ldb, Pointer.to(zero), C, ldc);
		}
	}

	public static void conv2d_backward_data(MatrixObject filter, MatrixObject dout,
			MatrixObject output, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor dyDesc = null;
		cudnnTensorDescriptor dxDesc = null;
		cudnnFilterDescriptor wDesc = null;
		cudnnConvolutionDescriptor convDesc = null;
		
		Pointer workSpace = null;
		long sizeInBytes = 0;
		try {
			// Allocate descriptors
			wDesc = allocateFilterDescriptor(K, C, R, S);
			dyDesc = allocateTensorDescriptor(N, K, P, Q);
			dxDesc = allocateTensorDescriptor(N, C, H, W);
			
			// Allocate data
			Pointer w = ((JCudaObject)filter.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer dx = ((JCudaObject)output.getGPUObject()).jcudaDenseMatrixPtr; 

			alpha = pointerTo(1.0); // TODO
			beta = pointerTo(0.0f);
			
			int padding [] = { pad_h, pad_w }; 
			int strides [] = { stride_h, stride_w };
			convDesc = allocateConvolutionDescriptor(padding, strides);
			long sizeInBytesArray[] = { 0 };
			
			// TODO: Select the best algorithm depending on the data and supported CUDA
			int algo = jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			workSpace = new Pointer();
			cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
					wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytesArray);
			
			int status = cudnnConvolutionBackwardData(cudnnHandle, alpha, wDesc, w, 
					dyDesc, dy, convDesc, algo, workSpace, sizeInBytes, beta, dxDesc, dx);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnConvolutionBackwardData: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(dyDesc != null)
				cudnnDestroyTensorDescriptor(dyDesc);
			if(dxDesc != null)
				cudnnDestroyTensorDescriptor(dxDesc);
			if(wDesc != null)
				cudnnDestroyFilterDescriptor(wDesc);
			
			if(convDesc != null)
				cudnnDestroyConvolutionDescriptor(convDesc);
			
			if(workSpace != null && sizeInBytes != 0)
				cudaFree(workSpace);
		}
	}
	
	/**
	 * performs maxpooling on GPU by exploiting cudnnPoolingForward(...)
	 * @param image
	 * @param outputBlock
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
	 * @throws DMLRuntimeException
	 */
	public static void maxpooling(MatrixObject image,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xDesc = null;
		cudnnTensorDescriptor yDesc = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			// Allocate descriptors
			yDesc = allocateTensorDescriptor(N, C, P, Q);
			xDesc = allocateTensorDescriptor(N, C, H, W);
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			
			// Allocate data
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr; 
			Pointer y = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr; 
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
			
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(yDesc != null)
				cudnnDestroyTensorDescriptor(yDesc);
			if(xDesc != null)
				cudnnDestroyTensorDescriptor(xDesc);
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);
		}
	}
	
	/**
	 * performs maxpoolingBackward on GPU by exploiting cudnnPoolingBackward(...)
	 * @param image
	 * @param dout			delta matrix, output of previous layer
	 * @param outputBlock
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
	 * @throws DMLRuntimeException
	 */
	public static void maxpooling_backward(MatrixObject image, MatrixObject dout,
			MatrixObject outputBlock, int N, int C, int H, int W, int K, int R,
			int S, int pad_h, int pad_w, int stride_h, int stride_w, int P,
			int Q) throws DMLRuntimeException {
		Pointer alpha = null;
		Pointer beta = null;
		cudnnTensorDescriptor xDesc = null;
		cudnnTensorDescriptor yDesc = null;
		cudnnTensorDescriptor dyDesc = null;
		cudnnTensorDescriptor dxDesc = null;
		cudnnPoolingDescriptor poolingDesc = null;

		try {
			// Allocate descriptors
			xDesc = allocateTensorDescriptor(N, C, H, W);
			yDesc = allocateTensorDescriptor(N, C, P, Q);
			dxDesc = allocateTensorDescriptor(N, C, H, W);
			dyDesc = allocateTensorDescriptor(N, C, P, Q);
			
			poolingDesc = allocatePoolingDescriptor(R, S, pad_h, pad_w, stride_h, stride_w);
			
			// Calling PoolForward first, y is one of the inputs for poolBackward
			// TODO: Remove calling poolForward after necessary changes at language level for poolBackward
			Pointer y = new Pointer();
			long numBytes = N*C*P*Q*Sizeof.DOUBLE;
			cudaMalloc(y, numBytes);
			
			// Allocate data
			Pointer x = ((JCudaObject)image.getGPUObject()).jcudaDenseMatrixPtr;
			Pointer dx = ((JCudaObject)outputBlock.getGPUObject()).jcudaDenseMatrixPtr;
			Pointer dy = ((JCudaObject)dout.getGPUObject()).jcudaDenseMatrixPtr;
			
			alpha = pointerTo(1.0);
			beta = pointerTo(0.0f);
			
			int status = cudnnPoolingForward(cudnnHandle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingForward before cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			
			status = cudnnPoolingBackward(cudnnHandle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
			
			if(status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
				throw new DMLRuntimeException("Could not executed cudnnPoolingBackward: " + jcuda.jcudnn.cudnnStatus.stringFor(status));
			}
			
			cudaFree(y);
		}
		finally {
			if(alpha != null)
				cudaFree(alpha);
			if(beta != null)
				cudaFree(beta);
			if(yDesc != null)
				cudnnDestroyTensorDescriptor(yDesc);
			if(xDesc != null)
				cudnnDestroyTensorDescriptor(xDesc);
			if(dyDesc != null)
				cudnnDestroyTensorDescriptor(dyDesc);
			if(dxDesc != null)
				cudnnDestroyTensorDescriptor(dxDesc);
			if(poolingDesc != null)
				cudnnDestroyPoolingDescriptor(poolingDesc);	
		}	
	}
	public static boolean isInSparseFormat(MatrixObject mo) {
		if(mo.getGPUObject() != null && mo.getGPUObject().isAllocated())
			return mo.getGPUObject().isInSparseFormat();
		return MatrixBlock.evalSparseFormatInMemory(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
	}
	
	/**
	 * Performs elementwise matrix-scalar operation specified by op
	 * 
	 * @param ec
	 * @param in
	 * @param outputName
	 * @param isInputTransposed
	 * @param op
	 * @throws DMLRuntimeException
	 */
	public static void bincellOp(ExecutionContext ec, MatrixObject in, String outputName, boolean isInputTransposed, ScalarOperator op) throws DMLRuntimeException {
		double constant = op.getConstant();
		boolean isCUDALibAvailable = (op.fn instanceof Multiply 
				|| (op.fn instanceof Divide && op instanceof RightScalarOperator && constant != 0)) && !isSparseAndEmpty(in);
		if(!isCUDALibAvailable) {
			if(constant == 0) {
				if(op.fn instanceof Plus || (op.fn instanceof Minus && op instanceof RightScalarOperator) || op.fn instanceof Or) {
					deviceCopy(ec, in, outputName, isInputTransposed);
				}
				else if(op.fn instanceof Multiply || op.fn instanceof And) {
					setOutputToConstant(ec, 0.0, outputName);
				}
				else if(op.fn instanceof Power) {
					setOutputToConstant(ec, 1.0, outputName);
				}
				else if(op.fn instanceof Divide && isSparseAndEmpty(in)) {
					setOutputToConstant(ec, Double.NaN, outputName);
				}
				else if(op.fn instanceof Divide) {
					//For division, IEEE 754 defines x/0.0 as INFINITY and 0.0/0.0 as NaN.
					compareAndSet(ec, in, outputName, 0.0, 1e-6, Double.NaN, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
				}
				else {
					// TODO: Potential to optimize
					launchBinCellOpKernel(ec, in, outputName, isInputTransposed, op);
				}
			}
			else if(constant == 1.0 && op.fn instanceof Or) {
				setOutputToConstant(ec, 1.0, outputName);
			}
			else if(constant == 1.0 && (op.fn instanceof And || op.fn instanceof Power)) {
				deviceCopy(ec, in, outputName, isInputTransposed);
			}
			else {
				launchBinCellOpKernel(ec, in, outputName, isInputTransposed, op);
			}
		}
		else {
			double alpha = 0;
			if(op.fn instanceof Multiply) {
				alpha = op.getConstant();
			}
			else if(op.fn instanceof Divide && op instanceof RightScalarOperator) {
				alpha = Math.pow((double)op.getConstant(), -1.0);
			}
			else {
				throw new DMLRuntimeException("Unsupported op");
			}
			
			// TODO: Performance optimization: Call cublasDaxpy if(in.getNumRows() == 1 || in.getNumColumns() == 1) 
			// C = alpha* op( A ) + beta* op ( B )
			dgeam(ec, in, in, outputName, isInputTransposed, isInputTransposed, alpha, 0.0);
		}
	}
	
	/**
	 * Utility to launch binCellScalarOp kernel
	 * 
	 * @param ec
	 * @param in
	 * @param outputName
	 * @param isInputTransposed
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void launchBinCellOpKernel(ExecutionContext ec, MatrixObject in, String outputName, boolean isInputTransposed, 
			ScalarOperator op) throws DMLRuntimeException {
		if(isInputTransposed)
			throw new DMLRuntimeException("Transposing the input is not supported");
		
		int rlenA = (int) in.getNumRows();
		int clenA = (int) in.getNumColumns();
		if(isInSparseFormat(in)) {
			// TODO: FIXME: Implement sparse binCellSparseScalarOp kernel
			((JCudaObject)in.getGPUObject()).sparseToDense();
		}
		Pointer A = ((JCudaObject)in.getGPUObject()).jcudaDenseMatrixPtr;
		double scalar = op.getConstant();
		MatrixObject out = ec.getMatrixObject(outputName);
		ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
	    Pointer C = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
	    int isLeftScalar = (op instanceof LeftScalarOperator) ? 1 : 0;
	    // Invokes binCellScalarOp(double* A, double scalar, double* C, int rlenA, int clenA, int op, int isLeftScalar)
		kernels.launchKernel("binCellScalarOp",
				ExecutionConfig.getConfigForSimpleMatrixOperations(rlenA, clenA), 
				A, scalar, C, rlenA, clenA, getBinaryOp(op.fn), isLeftScalar);
	}
	
	/**
	 * Utility to launch binCellOp kernel
	 * 
	 * @param ec
	 * @param in1
	 * @param in2
	 * @param outputName
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param op
	 * @throws DMLRuntimeException
	 */
	private static void launchBinCellOpKernel(ExecutionContext ec, MatrixObject in1, MatrixObject in2, 
			String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) throws DMLRuntimeException {
		boolean isSparse1 = isInSparseFormat(in1);
		boolean isEmpty1 = isSparseAndEmpty(in1);
		boolean isSparse2 = isInSparseFormat(in2);
		boolean isEmpty2 = isSparseAndEmpty(in2);
		if(isEmpty1) {
			// C = empty_in1 op in2 ==> becomes ==> C = 0.0 op in2
			bincellOp(ec, in2, outputName, isRightTransposed, new LeftScalarOperator(op.fn, 0.0));
		}
		else if(isEmpty2) {
			// C = in1 op empty_in2 ==> becomes ==> C = in1 op 0.0
			bincellOp(ec, in1, outputName, isLeftTransposed, new RightScalarOperator(op.fn, 0.0));
		}
		else {
			if(isSparse1) {
				// TODO: FIXME: Implement sparse binCellSparseOp kernel
				((JCudaObject)in1.getGPUObject()).sparseToDense();
			}
			Pointer A = ((JCudaObject)in1.getGPUObject()).jcudaDenseMatrixPtr;
			if(isSparse2) {
				// TODO: FIXME: Implement sparse binCellSparseOp kernel
				((JCudaObject)in2.getGPUObject()).sparseToDense();
			}
		    Pointer B = ((JCudaObject)in2.getGPUObject()).jcudaDenseMatrixPtr;
			
			int rlenA = (int) in1.getNumRows();
			int rlenB = (int) in2.getNumRows();
			int clenA = (int) in1.getNumColumns();
			int clenB = (int) in2.getNumColumns();
			MatrixObject out = ec.getMatrixObject(outputName);
			ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
		    Pointer C = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
		    // Invokes double* A, double* B, double* C, int maxRlen, int maxClen, int vectorAStatus, int vectorBStatus, int op
		    int maxRlen = Math.max(rlenA, rlenB);
		    int maxClen = Math.max(clenA, clenB);
		    int vecStatusA = getVectorStatus(in1);
		    int vecStatusB = getVectorStatus(in2);
			kernels.launchKernel("binCellOp",
					ExecutionConfig.getConfigForSimpleMatrixOperations(maxRlen, maxClen), 
					A, B, C, maxRlen, maxClen, vecStatusA, vecStatusB, getBinaryOp(op.fn));
		}
	}
	
	private static int getVectorStatus(MatrixObject in) {
		long rows = in.getNumRows();
		long cols = in.getNumColumns();
		if(cols == 1)
			return 1; 
		else if(rows == 1)
			return 2; 
		else
			return 0;
	}
	
	private static boolean isSparseAndEmpty(MatrixObject in1) {
		boolean isSparse1 = isInSparseFormat(in1);
		boolean isEmpty1 = isSparse1 && (((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);
		return isEmpty1;
	}
	
	private static void deviceCopy(ExecutionContext ec, MatrixObject src, String outputName, boolean isInputTransposed) throws DMLRuntimeException {
		if(!isInputTransposed)
			deviceCopy(ec, src, outputName);
		else
			transpose(ec, src, outputName);
	}
	
	/**
	 * Performs a deep device copy of input matrix
	 * 
	 * @param ec
	 * @param src
	 * @param outputName
	 * @throws DMLRuntimeException
	 */
	private static void deviceCopy(ExecutionContext ec, MatrixObject src, String outputName) throws DMLRuntimeException {
		if(isInSparseFormat(src)) {
			// TODO: FIXME: Implement sparse kernel
			((JCudaObject)src.getGPUObject()).sparseToDense();
		}
		Pointer srcPtr = ((JCudaObject)src.getGPUObject()).jcudaDenseMatrixPtr;
		MatrixObject out = ec.getMatrixObject(outputName);
		ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
	    Pointer destPtr = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
	    deviceCopy(srcPtr, destPtr, (int)src.getNumRows(), (int)src.getNumColumns());
	}
	
	private static void compareAndSet(ExecutionContext ec, MatrixObject in, String outputName, double compareVal,  double tolerance,
			double ifEqualsVal, double ifLessThanVal, double ifGreaterThanVal) throws DMLRuntimeException {
		if(isInSparseFormat(in)) {
			// TODO: FIXME: Implement sparse kernel
			((JCudaObject)in.getGPUObject()).sparseToDense();
		}
		Pointer A = ((JCudaObject)in.getGPUObject()).jcudaDenseMatrixPtr;
		MatrixObject out = ec.getMatrixObject(outputName);
		ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
	    Pointer ret = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
	    int rlen = (int) out.getNumRows();
	    int clen = (int) out.getNumColumns();
	    // out.getMatrixCharacteristics().setNonZeros(rlen*clen);
	    // compareAndSet(double* A,  double* ret, int rlen, int clen, double compareVal, double ifEqualsVal, double ifNotEqualsVal)
	    kernels.launchKernel("compareAndSet",
				ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
				A, ret, rlen, clen, compareVal, tolerance, ifEqualsVal, ifLessThanVal, ifGreaterThanVal);
	}
	
	/**
	 */
	private static void setOutputToConstant(ExecutionContext ec, double constant, String outputName) throws DMLRuntimeException {
		if(constant == 0) {
			// TODO: Create sparse empty block instead
		}
		MatrixObject out = ec.getMatrixObject(outputName);
		ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
	    Pointer A = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
	    int rlen = (int) out.getNumRows();
	    int clen = (int) out.getNumColumns();
//	    if(constant == 0) {
//	    	out.getMatrixCharacteristics().setNonZeros(0);
//	    }
//	    else {
//	    	out.getMatrixCharacteristics().setNonZeros(rlen*clen);
//	    }
	    // dense_matrix_set(double* A,  double scalar, int rlen, int clen)
	    kernels.launchKernel("dense_matrix_set",
				ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
				A, constant, rlen, clen);
	}
	
	/**
	 * Performs a deep copy of input device double pointer corresponding to matrix
	 * 
	 * @param src
	 * @param dest
	 * @param rlen
	 * @param clen
	 * @throws DMLRuntimeException
	 */
	private static void deviceCopy(Pointer src, Pointer dest, int rlen, int clen) throws DMLRuntimeException {
		kernels.launchKernel("dense_matrix_copy",
				ExecutionConfig.getConfigForSimpleMatrixOperations(rlen, clen),
				src, dest, rlen, clen);
	}
	
	/**
	 * Performs elementwise operation specified by op of two input matrices in1 and in2
	 * 
	 * 
	 * @param ec
	 * @param in1
	 * @param in2
	 * @param outputName
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param op
	 * @throws DMLRuntimeException
	 */
	public static void bincellOp(ExecutionContext ec, MatrixObject in1, MatrixObject in2, 
			String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) throws DMLRuntimeException {
		boolean isCUDALibAvailable = (op.fn instanceof Plus || op.fn instanceof Minus) && !isSparseAndEmpty(in1) && !isSparseAndEmpty(in2) && !isVector(in1) && !isVector(in2);
		if(!isCUDALibAvailable) {
			launchBinCellOpKernel(ec, in1, in2, outputName, isLeftTransposed, isRightTransposed, op);
		}
		else {
			double alpha;
			double beta;
			if(op.fn instanceof Plus) {
				alpha = 1.0;
				beta = 1.0;
			}
			else if(op.fn instanceof Minus) {
				alpha = 1.0;
				beta = -1.0;
			}
			else {
				throw new DMLRuntimeException("Unsupported op");
			}
			// C = alpha* op( A ) + beta* op ( B )
			dgeam(ec, in1, in2, outputName, isLeftTransposed, isRightTransposed, alpha, beta);
		}
	}
	
	private static boolean isVector(MatrixObject in) {
		return in.getNumRows() == 1 || in.getNumColumns() == 1;
	}
	
	// op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power, 
	// 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal, 
	// 11=min, 12=max, 13=and, 14=or, 15=log}
	private static int getBinaryOp(ValueFunction fn) throws DMLRuntimeException {
		if(fn instanceof Plus) return 0;
		else if(fn instanceof Minus) return 1;
		else if(fn instanceof Multiply) return 2;
		else if(fn instanceof Divide) return 3;
		else if(fn instanceof Power) return 4;
		else if(fn instanceof LessThan) return 5;
		else if(fn instanceof LessThanEquals) return 6;
		else if(fn instanceof GreaterThan) return 7;
		else if(fn instanceof GreaterThanEquals) return 8;
		else if(fn instanceof Equals) return 9;
		else if(fn instanceof NotEquals) return 10;
		else if(fn instanceof And) return 13;
		else if(fn instanceof Or) return 14;
		else if(fn instanceof Multiply2) return 2;
		else if(fn instanceof Power2) return 4;
		
		throw new DMLRuntimeException("The given value function is not supported:" + fn.getClass().getName());
	}
	
	/**
	 * Performs sparse and dense dgeam given two input matrices
	 * C = alpha* op( A ) + beta* op ( B )
	 * where op = transpose or not (specified by isLeftTransposed and isRightTransposed).
	 * 
	 * @param ec
	 * @param in1
	 * @param in2 
	 * @param outputName
	 * @param isLeftTransposed
	 * @param isRightTransposed
	 * @param alpha
	 * @param beta
	 * @throws DMLRuntimeException
	 */
	private static void dgeam(ExecutionContext ec, MatrixObject in1, MatrixObject in2, String outputName, 
			boolean isLeftTransposed, boolean isRightTransposed, double alpha, double beta) throws DMLRuntimeException {
		Pointer alphaPtr = pointerTo(alpha);
		Pointer betaPtr = pointerTo(beta);
		int transa = isLeftTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		int transb = isRightTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		int m = (int) in1.getNumRows();
		int n = (int) in1.getNumColumns();
		if(!isLeftTransposed && isRightTransposed) {
			m = (int) in1.getNumColumns();
			n = (int) in1.getNumRows();
		}
		int lda = isLeftTransposed ? n : m;
		int ldb = isRightTransposed ? n : m;
		int ldc = m;
		
		MatrixObject out = ec.getMatrixObject(outputName);
		boolean isSparse1 = isInSparseFormat(in1);
//		boolean isEmpty1 = isSparse1 && (((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);
		boolean isSparse2 = isInSparseFormat(in2);
//		boolean isEmpty2 = isSparse2 && (((JCudaObject)in2.getGPUObject()).jcudaSparseMatrixPtr.nnz == 0);
		
		// TODO: Implement sparse-dense matrix cublasDgeam kernel
		if(isSparse1 || isSparse2) {
			// Invoke cuSparse when either are in sparse format
	    	// Perform sparse-sparse dgeam
	    	if(!isInSparseFormat(in1)) {
				((JCudaObject)in1.getGPUObject()).denseToSparse();
			}
	    	CSRPointer A = ((JCudaObject)in1.getGPUObject()).jcudaSparseMatrixPtr;
	    	if(!isInSparseFormat(in2)) {
				((JCudaObject)in2.getGPUObject()).denseToSparse();
			}
			CSRPointer B = ((JCudaObject)in2.getGPUObject()).jcudaSparseMatrixPtr;
			
			ec.allocateGPUMatrixObject(outputName);
	    	
	    	CSRPointer C = CSRPointer.allocateForDgeam(cusparseHandle, A, B, m, n);
			((JCudaObject)out.getGPUObject()).setSparseMatrixCudaPointer(C);
			long sizeOfC = CSRPointer.estimateSize(C.nnz, out.getNumRows());
			out.getGPUObject().setDeviceModify(sizeOfC);
			JCusparse.cusparseDcsrgeam(cusparseHandle, m, n, alphaPtr, A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd, betaPtr, 
					B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd, 
					C.descr, C.val, C.rowPtr, C.colInd);
		}
		else {
			// Dense-Dense dgeam
			Pointer A = ((JCudaObject)in1.getGPUObject()).jcudaDenseMatrixPtr;
			Pointer B = ((JCudaObject)in2.getGPUObject()).jcudaDenseMatrixPtr;
			ec.getDenseMatrixOutputForGPUInstruction(outputName);	// Allocated the dense output matrix
		    Pointer C = ((JCudaObject)out.getGPUObject()).jcudaDenseMatrixPtr;
		    JCublas2.cublasDgeam(cublasHandle, transa, transb, m, n, alphaPtr, A, lda, betaPtr, B, ldb, C, ldc);
		}
	}
	
	/**
	 * Transposes the input matrix using cublasDgeam
	 * 
	 * @param ec
	 * @param in
	 * @param outputName
	 * @throws DMLRuntimeException
	 */
	public static void transpose(ExecutionContext ec, MatrixObject in, String outputName) throws DMLRuntimeException {
		// C = alpha* op( A ) + beta* op ( B )
		// = 1.0 * A^T + 0.0 * A^T
	    dgeam(ec, in, in, outputName, true, true, 1.0, 0.0);
	}
}

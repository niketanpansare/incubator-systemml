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
package org.apache.sysml.runtime.instructions.gpu.context;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsr2dense;
import static jcuda.jcusparse.JCusparse.cusparseDdense2csr;
import static jcuda.jcusparse.JCusparse.cusparseDnnz;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgemmNnz;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgeamNnz;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlockCOO;
import org.apache.sysml.runtime.matrix.data.SparseBlockCSR;
import org.apache.sysml.runtime.matrix.data.SparseBlockMCSR;
import org.apache.sysml.utils.Statistics;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseDirection;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparsePointerMode;
import jcuda.runtime.JCuda;

public class JCudaObject extends GPUObject {

	private static final Log LOG = LogFactory.getLog(JCudaObject.class.getName());

	/**
	 * Compressed Sparse Row (CSR) format for CUDA
	 * Generalized matrix multiply is implemented for CSR format in the cuSparse library
	 */
	public static class CSRPointer {
		
		public static cusparseMatDescr matrixDescriptor;
		
		/**
		 * @return Singleton default matrix descriptor object 
		 * 			(set with CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO)
		 */
		public static cusparseMatDescr getDefaultCuSparseMatrixDescriptor(){
			if (matrixDescriptor == null){
				// Code from JCuda Samples - http://www.jcuda.org/samples/JCusparseSample.java
				matrixDescriptor = new cusparseMatDescr();
				cusparseCreateMatDescr(matrixDescriptor);
				cusparseSetMatType(matrixDescriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatIndexBase(matrixDescriptor, CUSPARSE_INDEX_BASE_ZERO);
			}
			return matrixDescriptor;
		}
		
		private static final double ULTRA_SPARSITY_TURN_POINT = 0.0004;

		/**
		 * Default constructor to help with Factory method {@link #allocateCSRMatrix(long, long, long)}
		 */
		private CSRPointer() {
			val = new Pointer();
			rowPtr = new Pointer();
			colInd = new Pointer();
			allocateMatDescrPointer();
		}
		
		public long nnz;		/** Number of non zeroes	 									*/
		public Pointer val;		/** double array of non zero values 							*/
		public Pointer rowPtr;	/** integer array of start of all rows and end of last row + 1 	*/
		public Pointer colInd;	/** integer array of nnz values' column indices					*/
		public cusparseMatDescr descr;	/** descriptor of matrix, only CUSPARSE_MATRIX_TYPE_GENERAL supported	*/
		
		/** 
		 * Check for ultra sparsity
		 * @param rows
		 * @param cols
		 * @return
		 */
		public boolean isUltraSparse(int rows, int cols) {
			double sp = ((double)nnz/rows/cols);
			return sp<ULTRA_SPARSITY_TURN_POINT;
		}
		
		/**
		 * Initializes {@link #descr} to CUSPARSE_MATRIX_TYPE_GENERAL,
		 * the default that works for DGEMM.
		 */
		private void allocateMatDescrPointer() {			
			this.descr = getDefaultCuSparseMatrixDescriptor();
		}
		
		/**
		 * Estimate the size of a CSR matrix in GPU memory
		 * Size of pointers is not needed and is not added in
		 * @param nnz2	number of non zeroes
		 * @param rows	number of rows 
		 * @return
		 */
		public static long estimateSize(long nnz2, long rows) {
			long sizeofValArray = (Sizeof.DOUBLE) * nnz2;
			long sizeofRowPtrArray  = (Sizeof.INT) * (rows + 1);
			long sizeofColIndArray = (Sizeof.INT) * nnz2;
			long sizeofDescr = (Sizeof.INT) * 4;
			// From the CUSPARSE documentation, the cusparseMatDescr in native code is represented as: 
			// typedef struct {
			// 	cusparseMatrixType_t MatrixType;
			//	cusparseFillMode_t FillMode;
			//	cusparseDiagType_t DiagType;
			// 	cusparseIndexBase_t IndexBase;
			// } cusparseMatDescr_t;
			long tot = sizeofValArray + sizeofRowPtrArray + sizeofColIndArray + sizeofDescr;
			return tot;
		}
		
		/** 
		 * Factory method to allocate an empty CSR Sparse matrix on the GPU
		 * @param nnz2	number of non-zeroes
		 * @param rows 	number of rows
		 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
		 * @throws DMLRuntimeException 
		 */
		public static CSRPointer allocateEmpty(long nnz2, long rows) throws DMLRuntimeException {
			assert nnz2 > -1 : "Incorrect usage of internal API, number of non zeroes is less than 0 when trying to allocate sparse data on GPU";
			CSRPointer r = new CSRPointer();
			r.nnz = nnz2;
			if(nnz2 == 0) {
				// The convention for an empty sparse matrix is to just have an instance of the CSRPointer object
				// with no memory allocated on the GPU.
				return r;
			}
			ensureFreeSpace(Sizeof.DOUBLE * nnz2 + Sizeof.INT * (rows + 1) + Sizeof.INT * nnz2);
			r.val = allocate(Sizeof.DOUBLE * nnz2);
			r.rowPtr = allocate(Sizeof.INT * (rows + 1));
			r.colInd = allocate(Sizeof.INT * nnz2);
			return r;
		}
		
		/**
		 * Static method to copy a CSR sparse matrix from Host to Device
		 * @param dest	[input] destination location (on GPU)
		 * @param rows	number of rows
		 * @param nnz 	number of non-zeroes
		 * @param rowPtr	integer array of row pointers
		 * @param colInd	integer array of column indices
		 * @param values	double array of non zero values
		 * @return a {@link CSRPointer} instance that encapsulates the CSR matrix on GPU
		 */
		public static void copyToDevice(CSRPointer dest, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values) {
			CSRPointer r = dest;
			long t0 = System.nanoTime();
			r.nnz = nnz;
			cudaMemcpy(r.rowPtr, Pointer.to(rowPtr), (rows + 1) * Sizeof.INT, cudaMemcpyHostToDevice);
			cudaMemcpy(r.colInd, Pointer.to(colInd), nnz * Sizeof.INT, cudaMemcpyHostToDevice);
			cudaMemcpy(r.val, Pointer.to(values), nnz * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
			Statistics.cudaToDevTime.addAndGet(System.nanoTime()-t0);
			Statistics.cudaToDevCount.addAndGet(3);
		}
		
		/**
		 * Static method to copy a CSR sparse matrix from Device to host
		 * @param src	[input] source location (on GPU)
		 * @param rows	[input] number of rows
		 * @param nnz	[input] number of non-zeroes
		 * @param rowPtr	[output] pre-allocated integer array of row pointers of size (rows+1)
		 * @param colInd	[output] pre-allocated integer array of column indices of size nnz
		 * @param values	[output] pre-allocated double array of values of size nnz
		 */
		public static void copyToHost(CSRPointer src, int rows, long nnz, int[] rowPtr, int[] colInd, double[] values){
			CSRPointer r = src;
			long t0 = System.nanoTime();
			cudaMemcpy(Pointer.to(rowPtr), r.rowPtr, (rows + 1) * Sizeof.INT, cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(colInd), r.colInd, nnz * Sizeof.INT, cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(values), r.val, nnz * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			Statistics.cudaFromDevTime.addAndGet(System.nanoTime()-t0);
			Statistics.cudaFromDevCount.addAndGet(3);
		}

		
		// ==============================================================================================

		// The following methods estimate the memory needed for sparse matrices that are
		// results of operations on other sparse matrices using the cuSparse Library.
		// The operation is C = op(A) binaryOperation op(B), C is the output and A & B are the inputs
		// op = whether to transpose or not
		// binaryOperation = For cuSparse, +, - are *(matmul) are supported

		// From CuSparse Manual,
		// Since A and B have different sparsity patterns, cuSPARSE adopts a two-step approach
		// to complete sparse matrix C. In the first step, the user allocates csrRowPtrC of m+1
		// elements and uses function cusparseXcsrgeamNnz() to determine csrRowPtrC
		// and the total number of nonzero elements. In the second step, the user gathers nnzC
		//(number of nonzero elements of matrix C) from either (nnzC=*nnzTotalDevHostPtr)
		// or (nnzC=csrRowPtrC(m)-csrRowPtrC(0)) and allocates csrValC, csrColIndC of
		// nnzC elements respectively, then finally calls function cusparse[S|D|C|Z]csrgeam()
		// to complete matrix C.

		/**
		 * Allocate row pointers of m+1 elements
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param C			Output matrix
		 * @param rowsC			number of rows in C
		 * @throws DMLRuntimeException
		 */
		private static void step1AllocateRowPointers(cusparseHandle handle, CSRPointer C, int rowsC) throws DMLRuntimeException {
			cusparseSetPointerMode(handle, cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST);
			C.rowPtr = allocate(Sizeof.INT * (rowsC+1));
		}
		
		/**
		 * Determine total number of nonzero element for the cusparseDgeam  operation.
		 * This is done from either (nnzC=*nnzTotalDevHostPtr) or (nnzC=csrRowPtrC(m)-csrRowPtrC(0))
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param B			Sparse Matrix B on GPU
		 * @param C			Output Sparse Matrix C on GPU
		 * @param m			Rows in C
		 * @param n			Columns in C
		 * @throws DMLRuntimeException
		 */
		private static void step2GatherNNZGeam(cusparseHandle handle, CSRPointer A, CSRPointer B, CSRPointer C, int m, int n) throws DMLRuntimeException {
			int[] CnnzArray = { -1 };
			cusparseXcsrgeamNnz(handle, m, n,  
					A.descr, (int)A.nnz, A.rowPtr, A.colInd, 
					B.descr, (int)B.nnz, B.rowPtr, B.colInd, 
					C.descr, C.rowPtr, Pointer.to(CnnzArray));
			if (CnnzArray[0] != -1){
				C.nnz = CnnzArray[0];
			}
			else {
		        int baseArray[] = { 0 };
		        cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(m * Sizeof.INT), 1 * Sizeof.INT, cudaMemcpyDeviceToHost);
	            cudaMemcpy(Pointer.to(baseArray), C.rowPtr,								   1 * Sizeof.INT, cudaMemcpyDeviceToHost);
	            C.nnz = CnnzArray[0] - baseArray[0];
			}
		}

		/**
		 *	Determine total number of nonzero element for the cusparseDgemm operation.
		 *
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param transA	op - whether A is transposed
		 * @param B			Sparse Matrix B on GPU
		 * @param transB	op - whether B is transposed
		 * @param C			Output Sparse Matrix C on GPU
		 * @param m			Number of rows of sparse matrix op ( A ) and C
		 * @param n			Number of columns of sparse matrix op ( B ) and C
		 * @param k			Number of columns/rows of sparse matrix op ( A ) / op ( B )
		 * @throws DMLRuntimeException
		 */
		private static void step2GatherNNZGemm(cusparseHandle handle, CSRPointer A, int transA, CSRPointer B, int transB, CSRPointer C, int m, int n, int k) throws DMLRuntimeException {
			int[] CnnzArray = { -1 };
			if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) { 
				throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse"); 
			}
			cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, 
					A.descr, (int)A.nnz, A.rowPtr, A.colInd, 
					B.descr, (int)B.nnz, B.rowPtr, B.colInd, 
					C.descr, C.rowPtr, Pointer.to(CnnzArray));
			if (CnnzArray[0] != -1){
				C.nnz = CnnzArray[0];
			}
			else {
		        int baseArray[] = { 0 };
		        cudaMemcpy(Pointer.to(CnnzArray), C.rowPtr.withByteOffset(m * Sizeof.INT), 1 * Sizeof.INT, cudaMemcpyDeviceToHost);
	            cudaMemcpy(Pointer.to(baseArray), C.rowPtr,								   1 * Sizeof.INT, cudaMemcpyDeviceToHost);
	            C.nnz = CnnzArray[0] - baseArray[0];
			}
		}

		/**
		 * Allocate val and index pointers.
		 * 
		 * @param handle	a valid {@link cusparseHandle}
		 * @param C			Output sparse matrix on GPU
		 * @throws DMLRuntimeException
		 */
		private static void step3AllocateValNInd(cusparseHandle handle, CSRPointer C) throws DMLRuntimeException {
			C.val = allocate(Sizeof.DOUBLE * C.nnz);
			C.colInd = allocate(Sizeof.INT * C.nnz);
		}

		// ==============================================================================================


		/**
		 * Estimates the number of non zero elements from the results of a sparse cusparseDgeam operation
		 * C = a op(A) + b op(B)
		 * @param handle 	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param B			Sparse Matrix B on GPU
		 * @param m			Rows in A
		 * @param n			Columns in Bs
		 * @return
		 * @throws DMLRuntimeException
		 */
		public static CSRPointer allocateForDgeam(cusparseHandle handle, CSRPointer A, CSRPointer B, int m, int n) 
				throws DMLRuntimeException{
			if (A.nnz >= Integer.MAX_VALUE || B.nnz >= Integer.MAX_VALUE) { 
				throw new DMLRuntimeException("Number of non zeroes is larger than supported by cuSparse"); 
			}
			CSRPointer C = new CSRPointer();
			step1AllocateRowPointers(handle, C, m);
			step2GatherNNZGeam(handle, A, B, C, m, n);
			step3AllocateValNInd(handle, C);
			return C;
		}
		
		/**
		 * Estimates the number of non-zero elements from the result of a sparse matrix multiplication C = A * B
		 * and returns the {@link CSRPointer} to C with the appropriate GPU memory.
		 * @param handle	a valid {@link cusparseHandle}
		 * @param A			Sparse Matrix A on GPU
		 * @param transA	'T' if A is to be transposed, 'N' otherwise
		 * @param B			Sparse Matrix B on GPU
		 * @param transB	'T' if B is to be transposed, 'N' otherwise
		 * @param m			Rows in A
		 * @param n			Columns in B
		 * @param k			Columns in A / Rows in B
		 * @return
		 * @throws DMLRuntimeException
		 */
		public static CSRPointer allocateForMatrixMultiply(cusparseHandle handle, CSRPointer A, int transA, CSRPointer B, int transB, int m, int n, int k) 
				throws DMLRuntimeException{
			// Following the code example at http://docs.nvidia.com/cuda/cusparse/#cusparse-lt-t-gt-csrgemm and at
			// https://github.com/jcuda/jcuda-matrix-utils/blob/master/JCudaMatrixUtils/src/test/java/org/jcuda/matrix/samples/JCusparseSampleDgemm.java
			CSRPointer C = new CSRPointer();
			step1AllocateRowPointers(handle, C, m);
			step2GatherNNZGemm(handle, A, transA, B, transB, C, m, n, k);
			step3AllocateValNInd(handle, C);
			return C;
		}
		
		/**
		 * Copies this CSR matrix on the GPU to a dense column-major matrix
		 * on the GPU. This is a temporary matrix for operations such as 
		 * cusparseDcsrmv.
		 * Since the allocated matrix is temporary, bookkeeping is not updated.
		 * The caller is responsible for calling "free" on the returned Pointer object
		 * @param cusparseHandle	a valid {@link cusparseHandle}
		 * @param cublasHandle 		a valid {@link cublasHandle}
		 * @param rows		number of rows in this CSR matrix
		 * @param cols		number of columns in this CSR matrix
		 * @return			A {@link Pointer} to the allocated dense matrix (in column-major format)
		 * @throws DMLRuntimeException
		 */
		public Pointer toColumnMajorDenseMatrix(cusparseHandle cusparseHandle, cublasHandle cublasHandle, int rows, int cols) throws DMLRuntimeException {
			long size = rows * cols * Sizeof.DOUBLE;
			Pointer A = JCudaObject.allocate(size);
			// If this sparse block is empty, the allocated dense matrix, initialized to zeroes, will be returned.
			if (val != null && rowPtr != null && colInd != null && nnz > 0) {
				// Note: cusparseDcsr2dense method cannot handle empty blocks
				cusparseDcsr2dense(cusparseHandle, rows, cols, descr, val, rowPtr, colInd, A, rows);
			} else {
				LOG.warn("in CSRPointer, the values array, row pointers array or column indices array was null");
			}
			JCuda.cudaDeviceSynchronize();
			return A;
		}
		
		/**
		 * Calls cudaFree on the allocated {@link Pointer} instances
		 */
		public void deallocate() {
			cudaFree(val);
			cudaFree(rowPtr);
			cudaFree(colInd);
		}
	};
	
	public Pointer jcudaDenseMatrixPtr = null;		/** Pointer to dense matrix */
	public CSRPointer jcudaSparseMatrixPtr = null;	/** Pointer to sparse matrix */

	public long numBytes;

	JCudaObject(MatrixObject mat2) {
		super(mat2);
	}
	
	/**
	 * Allocates temporary space on the device.
	 * Does not update bookkeeping.
	 * The caller is responsible for freeing up after usage.
	 * @param size
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Pointer allocate(long size) throws DMLRuntimeException{
		Pointer A = new Pointer();
		ensureFreeSpace(size);
		long t0 = System.nanoTime();
		cudaMalloc(A, size);
		// Set all elements to 0 since newly allocated space will contain garbage
		cudaMemset(A, 0, size);
		Statistics.cudaAllocTime.getAndAdd(System.nanoTime() - t0);
		Statistics.cudaAllocCount.getAndAdd(1);
		return A;
	}

	/**
	 * Allocates a sparse and empty {@link JCudaObject}
	 * This is the result of operations that are both non zero matrices.
	 * @throws DMLRuntimeException
	 */
	public void allocateSparseAndEmpty() throws DMLRuntimeException{
		setSparseMatrixCudaPointer(CSRPointer.allocateEmpty(0, mat.getNumRows()));
		setDeviceModify(0);
	}


	/**
	 * Allocates a dense matrix of size obtained from the attached matrix metadata
	 * and fills it up with a single value
	 * @param v value to fill up the dense matrix
	 * @throws DMLRuntimeException
	 */
	public void allocateAndFillDense(double v) throws DMLRuntimeException {
		long rows = mat.getNumRows();
		long cols = mat.getNumColumns();
		int numElems = (int) (rows * cols);
		int size = numElems * Sizeof.DOUBLE;
		setDenseMatrixCudaPointer(allocate(size));
		setDeviceModify(size);
		// The "fill" kernel is called which treats the matrix "jcudaDensePtr" like a vector and fills it with value "v"
		LibMatrixCUDA.kernels.launchKernel("fill", ExecutionConfig.getConfigForSimpleVectorOperations(numElems), jcudaDenseMatrixPtr, v, numElems);
	}


	/**
	 * If this {@link JCudaObject} is sparse and empty
	 * Being allocated is a prerequisite to being sparse and empty.
	 * @return
	 */
	public boolean isSparseAndEmpty() {
		boolean isSparseAndAllocated = isAllocated && LibMatrixCUDA.isInSparseFormat(mat);
		boolean isEmptyAndSparseAndAllocated = isSparseAndAllocated && jcudaSparseMatrixPtr.nnz == 0;
		return isEmptyAndSparseAndAllocated;
	}

	/**
	 * Allocate necessary memory on the GPU for this {@link JCudaObject} instance.
	 * @param isInput if the block is input, isSparse argument is ignored
	 * @param isSparse if the block is sparse
	 * @throws DMLRuntimeException
	 */
	private void prepare(boolean isInput, boolean isSparse) throws DMLRuntimeException {
		if(jcudaDenseMatrixPtr != null || jcudaSparseMatrixPtr != null) {
			// Already allocated on GPU and expected to be in sync
		}
		else {
			if(isInput) {
				copyFromHostToDevice();
			}
			else {
				mat.setDirty(true);
				// Don't copy just allocate
				if (isSparse){
					long sparseSize = CSRPointer.estimateSize(mat.getNnz(), mat.getNumRows());
					ensureFreeSpace(sparseSize);
					allocateMemoryOnDevice(-1);
				} else { 	// Dense block, size = numRows * numCols
					int size = (int) (mat.getNumRows() * mat.getNumColumns());
					ensureFreeSpace(Sizeof.DOUBLE * size);
					allocateMemoryOnDevice(size);
				}
				synchronized(evictionLock) {
					GPUContext.allocatedPointers.add(this);
				}
			}
		}
		numLocks.addAndGet(1);
	}
	
	@Override
	public void acquireDeviceRead() throws DMLRuntimeException {
		prepare(true, false);
		if(!isAllocated) 
			throw new DMLRuntimeException("Expected device data to be allocated");
	}
	
	@Override
	public void acquireDeviceModifyDense() throws DMLRuntimeException {
		prepare(false, false); 
		isDeviceCopyModified = true;
		if(!isAllocated) 
			throw new DMLRuntimeException("Expected device data to be allocated");
	}
	
	@Override
	public void acquireDeviceModifySparse() throws DMLRuntimeException {
		isInSparseFormat = true;
		prepare(false, true);
		isDeviceCopyModified = true;
		if(!isAllocated) 
			throw new DMLRuntimeException("Expected device data to be allocated");
	}
	
	@Override
	public void acquireHostRead() throws CacheException {
		if(isAllocated) {
			try {
				if(isDeviceCopyModified) {
					copyFromDeviceToHost();
				}
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
		else {
			throw new CacheException("Cannot perform acquireHostRead as the GPU data is not allocated:" + mat.getVarName());
		}
	}
	
	@Override
	public void acquireHostModify() throws CacheException {
		if(isAllocated) {
			try {
				if(isDeviceCopyModified) {
					throw new DMLRuntimeException("Potential overwrite of GPU data");
					// copyFromDeviceToHost();
				}
				clearData();
			} catch (DMLRuntimeException e) {
				throw new CacheException(e);
			}
		}
	}
	
	/**
	 * updates the locks depending on the eviction policy selected
	 * @throws CacheException if there is no locked GPU Object
	 */
	private void updateReleaseLocks() throws CacheException {
		if(numLocks.addAndGet(-1) < 0) {
            throw new CacheException("Redundant release of GPU object");
		}
		if(evictionPolicy == EvictionPolicy.LRU) {
            timestamp.set(System.nanoTime());
		}
		else if(evictionPolicy == EvictionPolicy.LFU) {
            timestamp.addAndGet(1);
		}
		else if(evictionPolicy == EvictionPolicy.MIN_EVICT) {
            // Do Nothing
		}
		else {
            throw new CacheException("The eviction policy is not supported:" + evictionPolicy.name());
		}
	}
	
	/**
	 * releases input allocated on GPU
	 * @throws CacheException if data is not allocated
	 */
	public void releaseInput() throws CacheException {
		updateReleaseLocks();
		if(!isAllocated)
			throw new CacheException("Attempting to release an input before allocating it");
	}
	
	/**
	 * releases output allocated on GPU
	 * @throws CacheException if data is not allocated
	 */
	public void releaseOutput() throws CacheException {
		updateReleaseLocks();
		isDeviceCopyModified = true;
		if(!isAllocated)
			throw new CacheException("Attempting to release an output before allocating it");
	}

	@Override
	void allocateMemoryOnDevice(int numElemToAllocate) throws DMLRuntimeException {
		if(jcudaDenseMatrixPtr == null && jcudaSparseMatrixPtr == null) {
			long start = System.nanoTime();
			if(numElemToAllocate == -1 && LibMatrixCUDA.isInSparseFormat(mat)) {
				jcudaSparseMatrixPtr = CSRPointer.allocateEmpty(mat.getNnz(), mat.getNumRows()); 
				numBytes = CSRPointer.estimateSize(mat.getNnz(), mat.getNumRows());
				JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(-numBytes);
				isInSparseFormat = true;
				//throw new DMLRuntimeException("Sparse format not implemented");
			} else if(numElemToAllocate == -1) {
				// Called for dense input
				jcudaDenseMatrixPtr = new Pointer();
				numBytes = mat.getNumRows()*mat.getNumColumns()*Sizeof.DOUBLE;
				cudaMalloc(jcudaDenseMatrixPtr, numBytes);
				JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(-numBytes);
			}
			else {
				// Called for dense output
				jcudaDenseMatrixPtr = new Pointer();
				numBytes = numElemToAllocate*Sizeof.DOUBLE;
				cudaMalloc(jcudaDenseMatrixPtr,  numBytes);
				JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(-numBytes);
			}
			
			Statistics.cudaAllocTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaAllocCount.addAndGet(1);

		}
		isAllocated = true;
	}
	
	@Override
	public void setDeviceModify(long numBytes) {
		this.numLocks.addAndGet(1);
		this.numBytes = numBytes;
		JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(-numBytes);
	}

	@Override
	void deallocateMemoryOnDevice() {
		if(jcudaDenseMatrixPtr != null) {
			long start = System.nanoTime();
			cudaFree(jcudaDenseMatrixPtr);
			JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(numBytes);
			Statistics.cudaDeAllocTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaDeAllocCount.addAndGet(1);
		}
		if (jcudaSparseMatrixPtr != null) {
			long start = System.nanoTime();
			jcudaSparseMatrixPtr.deallocate();
			JCudaContext.availableNumBytesWithoutUtilFactor.addAndGet(numBytes);
			Statistics.cudaDeAllocTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaDeAllocCount.addAndGet(1);
		}
		jcudaDenseMatrixPtr = null;
		jcudaSparseMatrixPtr = null;
		isAllocated = false;
		numLocks.set(0);
	}
	
	/** 
	 * Thin wrapper over {@link #evict(long)}
	 * @param size
	 * @throws DMLRuntimeException
	 */
	static void ensureFreeSpace(long size) throws DMLRuntimeException {
		if(size >= getAvailableMemory()) {
			evict(size);
		}
	}
	
	@Override
	void copyFromHostToDevice() 
		throws DMLRuntimeException 
	{
		printCaller();
		long start = System.nanoTime();
		
		MatrixBlock tmp = mat.acquireRead();
		if(tmp.isInSparseFormat()) {
			
			int rowPtr[] = null;
			int colInd[] = null;
			double[] values = null;
			
			tmp.recomputeNonZeros();
			long nnz = tmp.getNonZeros();
			mat.getMatrixCharacteristics().setNonZeros(nnz);
			
			SparseBlock block = tmp.getSparseBlock();
			boolean copyToDevice = true;
			if(block == null && tmp.getNonZeros() == 0) {
//				// Allocate empty block --> not necessary
//				// To reproduce this, see org.apache.sysml.test.integration.applications.dml.ID3DMLTest
//				rowPtr = new int[0];
//				colInd = new int[0];
//				values = new double[0];
				copyToDevice = false;
			}
			else if(block == null && tmp.getNonZeros() != 0) {
				throw new DMLRuntimeException("Expected CP sparse block to be not null.");
			}
			else {
				// CSR is the preferred format for cuSparse GEMM
				// Converts MCSR and COO to CSR
				SparseBlockCSR csrBlock = null;
				if (block instanceof SparseBlockCSR){ 
					csrBlock = (SparseBlockCSR)block;
				} else if (block instanceof SparseBlockCOO) {
					// TODO - should we do this on the GPU using cusparse<t>coo2csr() ?
					long t0 = System.nanoTime();
					SparseBlockCOO cooBlock = (SparseBlockCOO)block;
					csrBlock = new SparseBlockCSR((int)mat.getNumRows(), cooBlock.rowIndexes(), cooBlock.indexes(), cooBlock.values());
					Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
					Statistics.cudaConversionCount.incrementAndGet();
				} else if (block instanceof SparseBlockMCSR) {
					long t0 = System.nanoTime();
					SparseBlockMCSR mcsrBlock = (SparseBlockMCSR)block;
					csrBlock = new SparseBlockCSR(mcsrBlock.getRows(), (int)mcsrBlock.size());
					Statistics.cudaConversionTime.addAndGet(System.nanoTime() - t0);
					Statistics.cudaConversionCount.incrementAndGet();
				} else {
					throw new DMLRuntimeException("Unsupported sparse matrix format for CUDA operations");
				}
				rowPtr = csrBlock.rowPointers();
				colInd = csrBlock.indexes();
				values = csrBlock.values();	
			}
			ensureFreeSpace(CSRPointer.estimateSize(mat.getNnz(), mat.getNumRows()));
			allocateMemoryOnDevice(-1);
			synchronized(evictionLock) {
				GPUContext.allocatedPointers.add(this);
			}
			if(copyToDevice) {
				CSRPointer.copyToDevice(jcudaSparseMatrixPtr, tmp.getNumRows(), tmp.getNonZeros(), rowPtr, colInd, values);
			}
			// throw new DMLRuntimeException("Sparse matrix is not implemented");
			// tmp.sparseToDense();
		}
		else {
			double[] data = tmp.getDenseBlock();
			
			if( data == null && tmp.getSparseBlock() != null )
				throw new DMLRuntimeException("Incorrect sparsity calculation");
			else if( data==null && tmp.getNonZeros() != 0 )
				throw new DMLRuntimeException("MatrixBlock is not allocated");
			else if( tmp.getNonZeros() == 0 )
				data = new double[tmp.getNumRows()*tmp.getNumColumns()];
			
			// Copy dense block
			ensureFreeSpace(Sizeof.DOUBLE * data.length);
			allocateMemoryOnDevice(data.length);
			synchronized(evictionLock) {
				GPUContext.allocatedPointers.add(this);
			}
			cudaMemcpy(jcudaDenseMatrixPtr, Pointer.to(data), mat.getNumRows()*mat.getNumColumns() * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		}
		
		mat.release();
		
		Statistics.cudaToDevTime.addAndGet(System.nanoTime()-start);
		Statistics.cudaToDevCount.addAndGet(1);
	}

	@Override
	protected void copyFromDeviceToHost() throws DMLRuntimeException {
		if (jcudaDenseMatrixPtr != null && jcudaSparseMatrixPtr != null){
			throw new DMLRuntimeException("Invalid state : JCuda dense/sparse pointer are both allocated");
		}

		if(jcudaDenseMatrixPtr != null) {
			printCaller();
			long start = System.nanoTime();
			MatrixBlock tmp = new MatrixBlock((int)mat.getNumRows(), (int)mat.getNumColumns(), false);
			tmp.allocateDenseBlock();
			double [] data = tmp.getDenseBlock();
			
			cudaMemcpy(Pointer.to(data), jcudaDenseMatrixPtr, data.length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			
			tmp.recomputeNonZeros();
			mat.acquireModify(tmp);
			mat.release();
			
			Statistics.cudaFromDevTime.addAndGet(System.nanoTime()-start);
			Statistics.cudaFromDevCount.addAndGet(1);
		}
		else if (jcudaSparseMatrixPtr != null){
			printCaller();
			if(!LibMatrixCUDA.isInSparseFormat(mat))
				throw new DMLRuntimeException("Block not in sparse format on host yet the device sparse matrix pointer is not null");

			if(this.isSparseAndEmpty()){
				MatrixBlock tmp = new MatrixBlock();	// Empty Block
				mat.acquireModify(tmp);
				mat.release();
			} else {
				long start = System.nanoTime();

				int rows = (int) mat.getNumRows();
				int cols = (int) mat.getNumColumns();
				int nnz = (int) jcudaSparseMatrixPtr.nnz;
				int[] rowPtr = new int[rows + 1];
				int[] colInd = new int[nnz];
				double[] values = new double[nnz];
				CSRPointer.copyToHost(jcudaSparseMatrixPtr, rows, nnz, rowPtr, colInd, values);

				SparseBlockCSR sparseBlock = new SparseBlockCSR(rowPtr, colInd, values, nnz);
				MatrixBlock tmp = new MatrixBlock(rows, cols, nnz, sparseBlock);
				mat.acquireModify(tmp);
				mat.release();
				Statistics.cudaFromDevTime.addAndGet(System.nanoTime() - start);
				Statistics.cudaFromDevCount.addAndGet(1);
			}
		}
		else {
			throw new DMLRuntimeException("Cannot copy from device to host as JCuda dense/sparse pointer is not allocated");
		}
		isDeviceCopyModified = false;
	}

	@Override
	protected long getSizeOnDevice() throws DMLRuntimeException {
		long GPUSize = 0;
		long rlen = mat.getNumRows();
		long clen = mat.getNumColumns();
		long nnz = mat.getNnz();

		if(LibMatrixCUDA.isInSparseFormat(mat)) {
			GPUSize = CSRPointer.estimateSize(nnz, rlen);
		}
		else {
			GPUSize = (Sizeof.DOUBLE) * (rlen * clen);
		}
		return GPUSize;
	}
	
	private String getClassAndMethod(StackTraceElement st) {
		String [] str = st.getClassName().split("\\.");
		return str[str.length - 1] + "." + st.getMethodName();
	}
	
	/**
	 * Convenience debugging method.
	 * Checks {@link JCudaContext#DEBUG} flag before printing to System.out
	 */
	private void printCaller() {
		if(JCudaContext.DEBUG) {
			StackTraceElement[] st = Thread.currentThread().getStackTrace();
			String ret = getClassAndMethod(st[1]);
			for(int i = 2; i < st.length && i < 7; i++) {
				ret += "->" + getClassAndMethod(st[i]);
			}
			System.out.println("CALL_STACK:" + ret);
		}
			
	}
	
	/**
	 * Convenience method to directly examine the Sparse matrix on GPU
	 */
	public CSRPointer getSparseMatrixCudaPointer() {
		return jcudaSparseMatrixPtr;
	}
	
	/**
	 * Convenience method to directly set the sparse matrix on GPU
	 * Needed for operations like {@link JCusparse#cusparseDcsrgemm(cusparseHandle, int, int, int, int, int, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, int, Pointer, Pointer, Pointer, cusparseMatDescr, Pointer, Pointer, Pointer)}
	 * @param jcudaSparseMatrixPtr
	 */
	public void setSparseMatrixCudaPointer(CSRPointer jcudaSparseMatrixPtr) {
		this.jcudaSparseMatrixPtr = jcudaSparseMatrixPtr;
		this.isAllocated = true;
		this.isInSparseFormat = true;
	}
	
	public void setDenseMatrixCudaPointer(Pointer densePtr){
		this.jcudaDenseMatrixPtr = densePtr;
		this.isAllocated = true;
		this.isInSparseFormat = false;
	}
	
	/**
	 * Converts the JCudaObject from dense to sparse format.
	 * 
	 * @throws DMLRuntimeException
	 */
	public void denseToSparse() throws DMLRuntimeException {
		cusparseHandle cusparseHandle = LibMatrixCUDA.cusparseHandle;
		if(cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = (int) mat.getNumRows();
		int cols = (int) mat.getNumColumns();
		
		if(jcudaDenseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated dense matrix before denseToSparse() call");
		
		// FIXME: denseToSparseUtil accepts column-major not row-major matrix, 
		// hence convertDensePtrFromColMajorToRowMajor required.
		convertDensePtrFromRowMajorToColumnMajor();
		setSparseMatrixCudaPointer(denseToSparseUtil(cusparseHandle, rows, cols, jcudaDenseMatrixPtr));
		cudaFree(jcudaDenseMatrixPtr);
		jcudaDenseMatrixPtr = null;
		// TODO: What if mat.getNnz() is -1 ?
		numBytes = CSRPointer.estimateSize(mat.getNnz(), rows);
	}


	private void transposeHelper(int m, int n, int lda, int ldc) throws DMLRuntimeException {
		if(jcudaDenseMatrixPtr == null || jcudaSparseMatrixPtr != null || !isAllocated) {
			throw new DMLRuntimeException("Internal Error in converting row major to column major or vice versa : either data is not allocated or internal state is corrupted");
		}

		Pointer alpha = LibMatrixCUDA.pointerTo(1.0);
		Pointer beta = LibMatrixCUDA.pointerTo(0.0);
		Pointer A = jcudaDenseMatrixPtr;
		Pointer C = JCudaObject.allocate(m*n* Sizeof.DOUBLE);

		// Transpose the matrix to get a dense matrix
		JCublas2.cublasDgeam(LibMatrixCUDA.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, alpha, A, lda, beta, new Pointer(), lda, C, ldc);
		jcudaDenseMatrixPtr = C;
		cudaFree(A);
	}

	private void convertDensePtrFromRowMajorToColumnMajor() throws DMLRuntimeException {
		int m = (int) mat.getNumRows();
		int n = (int) mat.getNumColumns();
		int lda = n;
		int ldc = m;

		transposeHelper(m, n, lda, ldc);
	}

	private void convertDensePtrFromColMajorToRowMajor() throws DMLRuntimeException {
		int n = (int) mat.getNumRows();
		int m = (int) mat.getNumColumns();
		int lda = n;
	    int ldc = m;

		transposeHelper(m, n, lda, ldc);
	}
	
	/**
	 * Convert sparse to dense (Performs transpose, use sparseToColumnMajorDense if the kernel can deal with column major format)
	 * 
	 * @throws DMLRuntimeException
	 */
	public void sparseToDense() throws DMLRuntimeException {
		if(jcudaSparseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");

		sparseToColumnMajorDense();
		convertDensePtrFromColMajorToRowMajor();
	}

	
	/**
	 * More efficient method to convert sparse to dense but returns dense in column major format
	 * 
	 * @throws DMLRuntimeException
	 */
	public void sparseToColumnMajorDense() throws DMLRuntimeException {
		if(jcudaSparseMatrixPtr == null || !isAllocated())
			throw new DMLRuntimeException("Expected allocated sparse matrix before sparseToDense() call");
		
		cusparseHandle cusparseHandle = LibMatrixCUDA.cusparseHandle;
		if(cusparseHandle == null)
			throw new DMLRuntimeException("Expected cusparse to be initialized");
		int rows = (int) mat.getNumRows();
		int cols = (int) mat.getNumColumns();
		setDenseMatrixCudaPointer(jcudaSparseMatrixPtr.toColumnMajorDenseMatrix(cusparseHandle, null, rows, cols));
		jcudaSparseMatrixPtr.deallocate();
		jcudaSparseMatrixPtr = null;
		numBytes = rows*cols*Sizeof.DOUBLE;
	}
	
	/**
	 * Convenience method to convert a CSR matrix to a dense matrix on the GPU
	 * Since the allocated matrix is temporary, bookkeeping is not updated.
	 * Caller is responsible for deallocating memory on GPU.
	 * Consider using denseToSparse() method if you are unsure.
	 * 
	 * @param rows
	 * @param cols
	 * @param densePtr	[in] dense matrix pointer on the GPU in row major
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static CSRPointer denseToSparseUtil(cusparseHandle cusparseHandle, int rows, int cols, Pointer densePtr) throws DMLRuntimeException {		
		cusparseMatDescr matDescr = CSRPointer.getDefaultCuSparseMatrixDescriptor();
		Pointer nnzPerRowPtr = new Pointer();
		Pointer nnzTotalDevHostPtr = new Pointer();
		
		ensureFreeSpace((rows + 1) * Sizeof.INT);
		
		long t1 = System.nanoTime();
		cudaMalloc(nnzPerRowPtr, cols * Sizeof.INT);
		cudaMalloc(nnzTotalDevHostPtr, Sizeof.INT);
		Statistics.cudaAllocTime.addAndGet(System.nanoTime() - t1);
		Statistics.cudaAllocCount.addAndGet(2);		
		
		// Output is in dense vector format, convert it to CSR
		cusparseDnnz(cusparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW, rows, cols, matDescr, densePtr, rows, nnzPerRowPtr, nnzTotalDevHostPtr);
		cudaDeviceSynchronize();
		int[] nnzC = {-1};
		
		long t2 = System.nanoTime();
		cudaMemcpy(Pointer.to(nnzC), nnzTotalDevHostPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
		Statistics.cudaFromDevTime.addAndGet(System.nanoTime() - t2);
		Statistics.cudaFromDevCount.addAndGet(2);		
		
		if (nnzC[0] == -1){
			throw new DMLRuntimeException("cusparseDnnz did not calculate the correct number of nnz from the sparse-matrix vector mulitply on the GPU");
		}
		
		CSRPointer C = CSRPointer.allocateEmpty(nnzC[0], rows);		
		cusparseDdense2csr(cusparseHandle, rows, cols, matDescr, densePtr, rows, nnzPerRowPtr, C.val, C.rowPtr, C.colInd);
		cudaDeviceSynchronize();

		cudaFree(nnzPerRowPtr);
		cudaFree(nnzTotalDevHostPtr);
		
		return C;
	}
	
	/**
	 * Gets the double array from GPU memory onto host memory and returns string.
	 * @param A Pointer to memory on device (GPU), assumed to point to a double array
	 * @param rows rows in matrix A
     * @param cols columns in matrix A
	 A*/
	private String debugString(Pointer A, long rows, long cols) {
		StringBuffer sb = new StringBuffer();
        int len = (int) (rows * cols);
		double[] tmp = new double[len];
		cudaMemcpy(Pointer.to(tmp), A, len*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        int k = 0;
		for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
			   sb.append(tmp[k]).append(' ');
               k++;
            }
            sb.append('\n');
		}
		return sb.toString();
	}
}

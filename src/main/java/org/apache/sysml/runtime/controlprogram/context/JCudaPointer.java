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

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import jcuda.Pointer;
import jcuda.Sizeof;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class JCudaPointer extends GPUPointer {
	
	public Pointer jcudaPointer = null;

	JCudaPointer(MatrixBlock mat) {
		super(mat);
	}

	@Override
	void allocateMemoryOnDevice() throws DMLRuntimeException {
		if(jcudaPointer == null) {
			jcudaPointer = new Pointer();
			if(mat.isInSparseFormat())
				throw new DMLRuntimeException("Sparse format not implemented");
			else
				cudaMalloc(jcudaPointer,  mat.getNumRows()*mat.getNumColumns()*Sizeof.DOUBLE);
		}
	}
	
	@Override
	void deallocateMemoryOnDevice() {
		if(jcudaPointer != null)
			cudaFree(jcudaPointer);
		jcudaPointer = null;
	}

	@Override
	void copyFromHostToDevice() throws DMLRuntimeException {
		if(jcudaPointer != null) {
			if(mat.isInSparseFormat())
				throw new DMLRuntimeException("Sparse format not implemented");
			else {
				double [] data = mat.getDenseBlock();
				if(data == null) {
					throw new DMLRuntimeException("MatrixBlock is not allocated");
				}
				cudaMemcpy(jcudaPointer, Pointer.to(data), mat.getNumRows()*mat.getNumColumns() * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
			}
		}
		else {
			throw new DMLRuntimeException("Cannot copy from host to device without allocating");
		}
		isDeviceCopyModified = false;
	}

	@Override
	public void copyFromDeviceToHost() throws DMLRuntimeException {
		if(mat.getNumRows() <= 0 || mat.getNumColumns() <= 0) {
			throw new DMLRuntimeException("Unknown matrix dimensions");
		}
		
		if(jcudaPointer != null) {
			if(mat.isInSparseFormat())
				throw new DMLRuntimeException("Sparse format not implemented");
			else {
				double [] data = mat.getDenseBlock();
				if(data == null) {
					mat.allocateDenseBlock();
					data = mat.getDenseBlock();
				}
				cudaMemcpy(Pointer.to(data), jcudaPointer, mat.getNumRows()*mat.getNumColumns() * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
			}
		}
		else {
			throw new DMLRuntimeException("Cannot copy from device to host as JCuda pointer is not allocated");
		}
		isDeviceCopyModified = false;
	}

	@Override
	public long getSizeOnDevice() throws DMLRuntimeException {
		long GPUSize = 0;
		boolean emptyBlock = (mat.getDenseBlock() == null && mat.getSparseBlock() == null);
		int rlen = mat.getNumRows();
		int clen = mat.getNumColumns();
//		long nonZeros = mat.getNonZeros();
		if (mat.isInSparseFormat() && !emptyBlock) {
//			GPUSize = (rlen + 1) * (long) (Integer.SIZE / Byte.SIZE) +
//					nonZeros * (long) (Integer.SIZE / Byte.SIZE) +
//					nonZeros * (long) (Double.SIZE / Byte.SIZE);
			throw new DMLRuntimeException("Sparse format not implemented");
		}
		else {
			int align = 0;
//			if (clen > 5120)
//				if (clen % 256 == 0)
//					align = 0;
//				else
//					align = 256; // in the dense case we use this for alignment
//			else
//				if (clen % 128 == 0)
//					align = 0;
//				else
//					align = 128; // in the dense case we use this for alignment
			GPUSize = (Sizeof.DOUBLE) * (long) (rlen * clen + align);
		}
		return GPUSize;
	}
}

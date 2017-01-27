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
package org.apache.sysml.udf.lib;

import java.io.IOException;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * Performs following operation:
 * # Computes the intersection ("meet") of equivalence classes for
 * # each row of A and B, excluding 0-valued cells.
 * # INPUT:
 * #   A, B = matrices whose rows contain that row's class labels;
 * #          for each i, rows A [i, ] and B [i, ] define two
 * #          equivalence relations on some of the columns, which
 * #          we want to intersect
 * #   A [i, j] == A [i, k] != 0 if and only if (j ~ k) as defined
 * #          by row A [i, ];
 * #   A [i, j] == 0 means that j is excluded by A [i, ]
 * #   B [i, j] is analogous
 * #   NOTE 1: Either nrow(A) == nrow(B), or exactly one of A or B
 * #   has one row that "applies" to each row of the other matrix.
 * #   NOTE 2: If ncol(A) != ncol(B), we pad extra 0-columns up to
 * #   max (ncol(A), ncol(B)).
 * # OUTPUT:
 * #   Both C and N have the same size as (the max of) A and B.
 * #   C = matrix whose rows contain class labels that represent
 * #       the intersection (coarsest common refinement) of the
 * #       corresponding rows of A and B.
 * #   C [i, j] == C [i, k] != 0 if and only if (j ~ k) as defined
 * #       by both A [i, ] and B [j, ]
 * #   C [i, j] == 0 if and only if A [i, j] == 0 or B [i, j] == 0
 * #       Additionally, we guarantee that non-0 labels in C [i, ]
 * #       will be integers from 1 to max (C [i, ]) without gaps.
 * #       For A and B the labels can be arbitrary.
 * #   N = matrix with class-size information for C-cells
 * #   N [i, j] = count of {C [i, k] | C [i, j] == C [i, k] != 0}
 *
 */
public class RowClassMeet extends PackageFunction {

	private static final long serialVersionUID = 1L;
	private Matrix CMat, NMat;
	private MatrixBlock A, B, C, N, NZ, A_full, B_full, tX_full;
	private int nr, nc;

	@Override
	public int getNumFunctionOutputs() {
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return CMat;
		else if(pos == 1)
			return NMat;
		else
			throw new RuntimeException("RowClassMeet produces only one output");
	}

	@Override
	public void execute() {
		try {
			A = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
			B = ((Matrix) getFunctionInput(1)).getMatrixObject().acquireRead();
			nr = Math.max(A.getNumRows(), B.getNumRows());
			nc = Math.max(A.getNumColumns(), B.getNumColumns());
			
			// Simplification:
			if(A.isInSparseFormat())
				A.sparseToDense();
			else if(A.getDenseBlock() == null)
				throw new RuntimeException("Empty input is not supported");
			if(B.isInSparseFormat())
				B.sparseToDense();
			else if(B.getDenseBlock() == null)
				throw new RuntimeException("Empty input is not supported");
			
			// C = matrix (0, rows = nr, cols = nc);
		    // N = matrix (0, rows = nr, cols = nc);
			// NZ = matrix (0, rows = nr, cols = nc);
			CMat = new Matrix( createOutputFilePathAndName( "TMP" ), nr, nc, ValueType.Double );
			C = new MatrixBlock(nr, nc, false);
			C.allocateDenseBlock();
			NMat = new Matrix( createOutputFilePathAndName( "TMP1" ), nr, nc, ValueType.Double );
			N = new MatrixBlock(nr, nc, false);
			N.allocateDenseBlock();
			NZ = new MatrixBlock(nr, nc, false);
			NZ.allocateDenseBlock();
		
//			if (nrow (A) == nr) {
//		        NZ [, 1 : ncol (A)] = ppred (A, 0, "!=");
//		        NZ [, 1 : ncol (B)] = NZ [, 1 : ncol (B)] * ppred (B, 0, "!=");
//		    } else {
//		        NZ [, 1 : ncol (B)] = ppred (B, 0, "!=");
//		        NZ [, 1 : ncol (A)] = NZ [, 1 : ncol (A)] * ppred (A, 0, "!=");
//		    }
			if(A.getNumRows() == nr) 
				operation1(A, B);
			else
				operation1(B, A);
			
			if(max(NZ) > 0) {
				// A_full = matrix (0, rows = nr, cols = nc);
				// B_full = matrix (0, rows = nr, cols = nc);
				// tX_full = matrix (0, rows = 3, cols = nc);
				A_full = new MatrixBlock(nr, nc, false);
				A_full.allocateDenseBlock();
				B_full = new MatrixBlock(nr, nc, false);
				B_full.allocateDenseBlock();
				tX_full = new MatrixBlock(3, nc, false);
				tX_full.allocateDenseBlock();
				
				// A_full [, 1 : ncol (A)] = NZ [, 1 : ncol (A)] * A
				operation2(A_full, A);
				// B_full [, 1 : ncol (B)] = NZ [, 1 : ncol (B)] * B;
				operation2(B_full, B);
				// tX_full [3, ] = t(seq (1, nc, 1));
				for(int j = 0; j < tX_full.getNumColumns(); j++) {
					set(tX_full, 2, j, j+1);
				}
				
				for(int rowID = 0; rowID < nr; rowID++) {
					if(sum(NZ, rowID) > 0) {
//						tX_full [1, ] = A_full [rowID, ];
//		                tX_full [2, ] = B_full [rowID, ];
						row_copy(tX_full, A_full, 0, rowID);
						row_copy(tX_full, B_full, 1, rowID);
						
						// TODO:
						
					}
				}
			}
			
			((Matrix) getFunctionInput(0)).getMatrixObject().release();
			((Matrix) getFunctionInput(1)).getMatrixObject().release();
		} catch (CacheException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		}
		
		
		try {
			C.recomputeNonZeros();
			C.examSparsity();
			CMat.setMatrixDoubleArray(C, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			N.recomputeNonZeros();
			N.examSparsity();
			NMat.setMatrixDoubleArray(N, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		} catch (IOException e) {
			throw new RuntimeException("Error while executing RowClassMeet", e);
		}
	}
	
	void row_copy(MatrixBlock dest, MatrixBlock src, int destRowIndex, int srcRowIndex) throws DMLRuntimeException {
		if(src.getNumColumns() != dest.getNumColumns())
			throw new DMLRuntimeException("The number of columns of src and dest should match");
		System.arraycopy(src, srcRowIndex*src.getNumColumns(), dest, destRowIndex*dest.getNumColumns(), src.getNumColumns());
	}
	
	// NZ [, 1 : ncol (A)] = ppred (A, 0, "!=");
	// NZ [, 1 : ncol (B)] = NZ [, 1 : ncol (B)] * ppred (B, 0, "!=");
	void operation1(MatrixBlock A, MatrixBlock B) {
		for(int i = 0; i < A.getNumRows(); i++) {
			for(int j = 0; j < A.getNumColumns(); j++) {
				set(NZ, i, j , ppred_not_eq_zero(A, i, j));
			}
		}
		for(int i = 0; i < B.getNumRows(); i++) {
			for(int j = 0; j < B.getNumColumns(); j++) {
				set(NZ, i, j , get(NZ, i, j)*ppred_not_eq_zero(A, i, j));
			}
		}
	}
	
	// A_full [, 1 : ncol (A)] = NZ [, 1 : ncol (A)] * A
	void operation2(MatrixBlock A_full, MatrixBlock A) {
		for(int i = 0; i < A.getNumRows(); i++) {
			for(int j = 0; j < A.getNumColumns(); j++) {
				set(A_full, i, j , get(NZ, i, j)*get(A, i, j));
			}
		}
	}
	
//	  X = t(removeEmpty (target = tX_full, margin = "cols", select = NZ [rowID, ]));
//    nx = nrow (X);
//    X = order (target = X, by = 1, decreasing = FALSE, index.return = FALSE);
//    X = order (target = X, by = 2, decreasing = FALSE, index.return = FALSE);
	MatrixBlock operation3(MatrixBlock tX_full, int rowID) {
		
		
		return null;
	}
	
	double ppred_not_eq_zero(MatrixBlock A, int i, int j) {
		return A.getDenseBlock()[i*A.getNumColumns()+j] != 0 ? 1 : 0;
	}
	
	void set(MatrixBlock A, int i, int j, double val) {
		A.getDenseBlock()[i*A.getNumColumns()+j] = val;
	}
	double  get(MatrixBlock A, int i, int j) {
		return A.getDenseBlock()[i*A.getNumColumns()+j];
	}
	
	double sum(MatrixBlock A, int rowID) {
		double sum = 0;
		for(int j = 0; j < A.getNumColumns(); j++) {
			sum += get(A, rowID, j);
		}
		return sum;
	}
	
	double max(MatrixBlock A) {
		double [] denseblk = A.getDenseBlock(); 
		double ret = denseblk[0];
		for(int i = 1; i < denseblk.length; i++)
			ret = Math.max(ret, denseblk[i]);
		return ret;
	}
}

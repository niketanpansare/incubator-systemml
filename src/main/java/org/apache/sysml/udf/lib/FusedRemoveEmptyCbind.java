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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * This external built-in function addresses following two common scenarios:
 * 1. cbind (cbind (cbind ( X1, X2 ), X3 ), X4)
 * 2. removeEmpty (target = cbind (cbind ( spagetize(X1), spagetize(X2) ), spagetize(X3) ), margin="rows", select=X4)
 * 
 * spagetize = function(matrix[double] X) return (matrix[double] Y) {
 *    Y = matrix (X1, rows=length(X1), cols=1)
 * }
 * 
 * The API of this external built-in function is as follows:
 * 
 * func = externalFunction(int numInputs, boolean performOnlyCBind, matrix[double] X1, matrix[double] X2,  matrix[double] X3, matrix[double] X4) return (matrix[double] out) 
 * implemented in (classname="org.apache.sysml.udf.lib.FusedRemoveEmptyCbind",exectype="mem");
 * 
 * Since this operator is very specific, we kept it as an external built-in function.
 * If there is huge interest in this functionality, we can later generalize both cbind and removeEmpty to accomodate this functionality and eliminate this external built-in function.
 */
public class FusedRemoveEmptyCbind extends PackageFunction {
	private static final long serialVersionUID = -4266180315672563097L;

	private Matrix ret;
	private MatrixBlock retMB;
	
	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return ret;
		else
			throw new RuntimeException("FusedRemoveEmptyCbind produces only one output");
	}

	@Override
	public void execute() {
		int numInputs = Integer.parseInt(((Scalar)getFunctionInput(0)).getValue());
		boolean performOnlyCBind = Boolean.parseBoolean(((Scalar)getFunctionInput(1)).getValue());
		if(!performOnlyCBind) {
			// Get the row indexes from the filter that are not NNZ
			int filterIndex = numInputs + 1;
			ArrayList<Integer> sortedRowsWithNNZ = new ArrayList<Integer>(computeRowsWithNNZ(filterIndex));
			Collections.sort(sortedRowsWithNNZ);
			numRetRows = sortedRowsWithNNZ.size()*numColsFilter;
			numRetCols = numInputs; // Since the inputs are spagetized
			
			// Allocate output
			allocateOutput();
			double [] retData = retMB.getDenseBlock();
			
			try {
				// Perform removeEmpty (target = cbind (cbind ( spagetize(X1), spagetize(X2) ), spagetize(X3) ), margin="rows", select=X4)
				int outputColIndex = 0;
				for(int inputID = 2; inputID < numInputs + 2; inputID++, outputColIndex++) {
					for(int unspagetizedOutputRowIndex = 0; unspagetizedOutputRowIndex < sortedRowsWithNNZ.size(); unspagetizedOutputRowIndex++) {
						int inputRowIndex = sortedRowsWithNNZ.get(unspagetizedOutputRowIndex);
						int outputRowOffset = unspagetizedOutputRowIndex*numColsFilter;
						MatrixBlock in = ((Matrix) getFunctionInput(inputID)).getMatrixObject().acquireRead();
						if(in.isInSparseFormat()) {
							Iterator<IJV> iter = in.getSparseBlockIterator(inputRowIndex, inputRowIndex+1);
							while(iter.hasNext()) {
								IJV ijv = iter.next();
								int outputRowIndex = outputRowOffset + ijv.getJ();
								retData[outputRowIndex*retMB.getNumColumns() + outputColIndex] = ijv.getV();
							}
						}
						else {
							double [] inData = in.getDenseBlock();
							if(inData != null) {
								int inputOffset = inputRowIndex*numColsFilter;
								for(int j = 0; j < numColsFilter; j++) {
									int outputRowIndex = outputRowOffset + j;
									retData[outputRowIndex*retMB.getNumColumns() + outputColIndex] = inData[inputOffset + j];
								}
							}
						}
						
						((Matrix) getFunctionInput(inputID)).getMatrixObject().release();
					}
				}
			} catch (CacheException e) {
				throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
			}
		}
		else {
			computeCbindOnlyOutputDimensions(numInputs);
			allocateOutput();
			performCBind(numInputs);
		}
		
		retMB.recomputeNonZeros();
		try {
			retMB.examSparsity();
			ret.setMatrixDoubleArray(retMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
		} catch (IOException e) {
			throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
		}	
	}
	
	long numRetRows; long numRetCols;
	private void computeCbindOnlyOutputDimensions(int numInputs) {
		try {
			numRetCols = 0;
			for(int inputID = 2; inputID < numInputs + 2; inputID++) {
				MatrixBlock in = ((Matrix) getFunctionInput(inputID)).getMatrixObject().acquireRead();
				numRetRows = in.getNumRows();
				numRetCols += in.getNumColumns();
				((Matrix) getFunctionInput(inputID)).getMatrixObject().release();
			}
		} catch (CacheException e) {
			throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
		}
	}
	
	int numRowsFilter = -1;
	int numColsFilter = -1;
	private HashSet<Integer> computeRowsWithNNZ(int filterIndex) {
		try {
			MatrixBlock filter = ((Matrix) getFunctionInput(filterIndex)).getMatrixObject().acquireRead();
			numRowsFilter = filter.getNumRows();
			numColsFilter = filter.getNumColumns();
			HashSet<Integer> rowsWithNNZ = new HashSet<Integer>();
			if(filter.isInSparseFormat()) {
				for(int i = 0; i < filter.getNumRows(); i++) {
					Iterator<IJV> iter = filter.getSparseBlockIterator(i, i+1);
					while(iter.hasNext()) {
						IJV ijv = iter.next();
						if(ijv.getV() != 0) {
							rowsWithNNZ.add(ijv.getI());
							break;
						}
					}
				}
			}
			else {
				double [] denseBlock = filter.getDenseBlock();
				if(denseBlock != null) {
					for(int i = 0; i < filter.getNumRows(); i++) {
						for(int j = 0; j < filter.getNumColumns(); j++) {
							if(denseBlock[i*filter.getNumColumns() + j] != 0) {
								rowsWithNNZ.add(i);
								break;
							}
						}
					}
				}
			}
			((Matrix) getFunctionInput(filterIndex)).getMatrixObject().release();
			return rowsWithNNZ;
		} catch (CacheException e) {
			throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
		}
	}
	
	// Performs cbind (cbind (cbind ( X1, X2 ), X3 ), X4)
	private void performCBind(int numInputs) {
		double [] retData = retMB.getDenseBlock();
		try {
			int startColumn = 0;
			for(int inputID = 2; inputID < numInputs + 2; inputID++) {
				MatrixBlock in = ((Matrix) getFunctionInput(inputID)).getMatrixObject().acquireRead();
				int inputNumCols = in.getNumColumns();
				if(in.isInSparseFormat()) {
					Iterator<IJV> iter = in.getSparseBlockIterator();
					while(iter.hasNext()) {
						IJV ijv = iter.next();
						int outputRowIndex = ijv.getI();
						int outputColIndex = ijv.getJ() + startColumn;
						retData[(int) (outputRowIndex*retMB.getNumColumns() + outputColIndex)] = ijv.getV();
					}
				}
				else {
					double [] denseBlock = in.getDenseBlock();
					if(denseBlock != null) {
						for(int i = 0; i < retMB.getNumRows(); i++) {
							for(int j = 0; j < inputNumCols; j++) {
								int outputColIndex = j + startColumn;
								retData[(int) (i*retMB.getNumColumns() + outputColIndex)] = denseBlock[i*inputNumCols + j];
							}
						}
					}
				}
				((Matrix) getFunctionInput(inputID)).getMatrixObject().release();
				startColumn += inputNumCols;
			}
		} catch (CacheException e) {
			throw new RuntimeException("Error while executing FusedRemoveEmptyCbind", e);
		}
	}
	
	private void allocateOutput() {
		String dir = createOutputFilePathAndName( "TMP" );
		ret = new Matrix( dir, numRetRows, numRetCols, ValueType.Double );
		retMB = new MatrixBlock((int) numRetRows, (int) numRetCols, false);
		retMB.allocateDenseBlock();
	}
	

}

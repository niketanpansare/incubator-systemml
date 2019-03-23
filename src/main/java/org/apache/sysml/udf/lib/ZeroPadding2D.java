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
import java.util.Random;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

// TODO: Move this from external builtin function to language-level builtin function
public class ZeroPadding2D extends PackageFunction {

	private static final long serialVersionUID = 1L;
	
	private Matrix output = null;
	private static Random rand = new Random();

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		return output;
	}

	@Override
	public void execute() {
		// input , int C, int Hin, int Win, int top_pad, int bottom_pad, int left_pad, int right_pad, int isForward
		final MatrixBlock in = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
		final int C = Integer.parseInt(((Scalar)getFunctionInput(1)).getValue());
		final int Hin = Integer.parseInt(((Scalar)getFunctionInput(2)).getValue());
		final int Win = Integer.parseInt(((Scalar)getFunctionInput(3)).getValue());
		final int top_pad = Integer.parseInt(((Scalar)getFunctionInput(4)).getValue());
		final int bottom_pad = Integer.parseInt(((Scalar)getFunctionInput(5)).getValue());
		final int left_pad = Integer.parseInt(((Scalar)getFunctionInput(6)).getValue());
		final int right_pad = Integer.parseInt(((Scalar)getFunctionInput(7)).getValue());
		final int isForward1 = Integer.parseInt(((Scalar)getFunctionInput(8)).getValue());
		final boolean isForward = isForward1 == 1 ? true : false;
		final int Hout = Hin+top_pad+bottom_pad;
		final int Wout = Win+left_pad+right_pad;
		final int N = in.getNumRows();
		output = new Matrix( "tmp_" + rand.nextLong(), N, C*Hout*Wout, ValueType.Double );
		final MatrixBlock outputMB = allocateDenseMatrixBlock(output);
		double [] outputArr = outputMB.getDenseBlockValues();
		try {
			for(int n = 0; n < N; n++) {
				for(int c = 0; c < C; c++) {
					for(int h = 0; h < Hin; h++) {
						int pad_col_offset = c*Hout*Wout + (h + top_pad)*Wout;
						int pad_offset = n*C*Hout*Wout + pad_col_offset;
						int col_offset = c*Hin*Win;
						int offset = n*C*Hin*Win + col_offset;
						for(int w = 0; w < Win; w++) {
							int padded_w = w + left_pad;
							if(isForward) {
								outputArr[pad_offset + padded_w] = in.getValue(n, col_offset + w);
							}
							else {
								outputArr[offset + w] = in.getValue(n, pad_col_offset + padded_w);
							}
						}
					}
				}
			}
			outputMB.setNonZeros(in.getNonZeros());
			output.setMatrixDoubleArray(outputMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} 
		catch (IOException e) {
			throw new DMLRuntimeException("Error occured while executing ZeroPadding2D external builtin function", e);
		}
		finally {
			((Matrix) getFunctionInput(0)).getMatrixObject().release();
		}
	}
	
	private static MatrixBlock allocateDenseMatrixBlock(Matrix mat) {
		int rows = (int) mat.getNumRows();
		int cols = (int) mat.getNumCols();
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		mb.allocateDenseBlock();
		return mb;
	}

}

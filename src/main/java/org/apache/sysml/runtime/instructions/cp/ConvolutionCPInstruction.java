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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class ConvolutionCPInstruction extends UnaryCPInstruction {
	
	private static final Log LOG = LogFactory.getLog(ConvolutionCPInstruction.class.getName());

	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();

	public ConvolutionCPInstruction(CPOperand in, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public static ConvolutionCPInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("reshape_col")
				|| opcode.equalsIgnoreCase("rotate180")
				|| opcode.equalsIgnoreCase("im2col")
				|| opcode.equalsIgnoreCase("col2im")
				|| opcode.equalsIgnoreCase("pooling_pre_reshape")
				|| opcode.equalsIgnoreCase("pooling_post_reshape")) {
			InstructionUtils.checkNumFields(parts, 14);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			out.split(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[2]));
			stride.add(new CPOperand(parts[3]));
			padding.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			input_shape.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			filter_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));

			return new ConvolutionCPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}

	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// acquire inputs
		MatrixBlock outputBlock = null;
		if (instOpcode.equalsIgnoreCase("im2col")
				|| instOpcode.equalsIgnoreCase("reshape_col")
				|| instOpcode.equalsIgnoreCase("rotate180")
				|| instOpcode.equalsIgnoreCase("col2im")
				|| instOpcode.equalsIgnoreCase("pooling_pre_reshape")
				|| instOpcode.equalsIgnoreCase("pooling_post_reshape")) {
			
			MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
			pad_h = getScalarInput(ec, _padding, 0);
			pad_w = getScalarInput(ec, _padding, 1);
			stride_h = getScalarInput(ec, _stride, 0);
			stride_w = getScalarInput(ec, _stride, 1);

			N = getScalarInput(ec, _input_shape, 0);
			C = getScalarInput(ec, _input_shape, 1);
			H = getScalarInput(ec, _input_shape, 2);
			W = getScalarInput(ec, _input_shape, 3);

			K = getScalarInput(ec, _filter_shape, 0);
			
			R = getScalarInput(ec, _filter_shape, 2);
			S = getScalarInput(ec, _filter_shape, 3);
			
			P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
			Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
			
			
			if (instOpcode.equalsIgnoreCase("im2col")) {
				checkHeightWidth(ec);
				checkInputDimensionForIm2col(matBlock);
				outputBlock = im2col(matBlock);
			}
			else if (instOpcode.equalsIgnoreCase("reshape_col")) {
				checkHeightWidth(ec);
				outputBlock = reshape_col(matBlock);
			}
			else if (instOpcode.equalsIgnoreCase("rotate180")) {
				checkHeightWidth(ec);
				outputBlock = rotate180(matBlock);
			}
			else if (instOpcode.equalsIgnoreCase("col2im")) {
				checkHeightWidth(ec);
				checkInputDimensionForCol2im(matBlock);
				outputBlock = col2im(matBlock);
			}
			else if (instOpcode.equalsIgnoreCase("pooling_pre_reshape")) {
				outputBlock = pooling_pre_reshape(matBlock);
			}
			else if (instOpcode.equalsIgnoreCase("pooling_post_reshape")) {
				outputBlock = pooling_post_reshape(matBlock);
			}
			else {
				throw new DMLRuntimeException("Unsupported op code " + instOpcode);
			}
		} else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), outputBlock);
	}
	
	int N; int C; int H; int W;
	int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
	int P; int Q;
	double[] outputArray; double[] inputArray; MatrixBlock input;
	long outNNZ = 0;
	// These are used to iterate
	int i;  int row;

	private void checkHeightWidth(ExecutionContext ec) throws DMLRuntimeException {
		int numChannelsInFilter = getScalarInput(ec, _filter_shape, 1);
		
		if (numChannelsInFilter != C) { 
			throw new DMLRuntimeException("The number of channels of input and filter should match");
		}
		if((W + 2 * pad_w - S) % stride_w != 0) {
			throw new DMLRuntimeException("The width does not work (Hint: (W + 2 * pad_w - S) % stride_w should be 0 [ ==> (" + W + "+" + " 2*" + pad_w + "-" +  S + ") % " + stride_w + "!= 0] ");
		}
		if((H + 2 * pad_h - R) % stride_h != 0) {
			throw new DMLRuntimeException("The height does not work (Hint: (H + 2 * pad_h - R) % stride_h should be 0 [ ==> (" + H + "+" + " 2*" + pad_h + "-" +  R + ") % " + stride_h + "!= 0] ");
		}
		if(H <= 0) {
			throw new DMLRuntimeException("Height of output patch should be zero");
		}
		if(Q <= 0) {
			throw new DMLRuntimeException("Width of output patch should be zero");
		}
	}
	
	private MatrixBlock init(int numRowsOutput, int numColsOutput) throws DMLRuntimeException {
		// int numColsOutput = N * P * Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		i = 0; row = 0; 
		return output;
	}
	
	// Reshape a matrix of dimension (1, N*C*P*Q) of dimension a 4D tensor of dimension (N, C, P, Q)
	private MatrixBlock pooling_post_reshape(MatrixBlock input) throws DMLRuntimeException {
		this.input = input;
		int numRowsOutput = N;
		int numColsOutput = C*P*Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		
		if(!input.isInSparseFormat()) {
			// TODO: Do in-place update
			double [] inputArray = input.getDenseBlock();
			for(int i = 0; i < inputArray.length; i++) {
				outputArray[i] = inputArray[i];
			}
			return output;			
		}
		
		if(input.getNumColumns() != N*C*P*Q || input.getNumRows() != 1) {
			throw new DMLRuntimeException("Incorrect input dimensions in pooling_post_reshape:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int p = 0; p < P; p++) {
					for (int q = 0; q < Q; q++) {
						outputArray[n*C*P*Q + c*P*Q + p*Q + q] = input.getValue(1, n*C*P*Q + c*P*Q + p*Q + q);
					}
				}
			}
		}
		
		return output;
	}

	// Reshape a 4D tensor of dimension (N, C, H, W) to a 4D tensor of dimension of dimension (N*C, 1, H, W)
	private MatrixBlock pooling_pre_reshape(MatrixBlock input) throws DMLRuntimeException {
		this.input = input;
		int numRowsOutput = N*C;
		int numColsOutput = H*W;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(input.getNumColumns() != C*H*W || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in pooling_pre_reshape:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {		
						if(inputArray != null)
							outputArray[(n*C + c)*H*W + h*H + w] = inputArray[n*C*H*W + c*H*W + h*W + w];
						else
							outputArray[(n*C + c)*H*W + h*H + w] = input.getValue(n, c*H*W + h*W + w);
					}
				}
			}
		}
		
		return output;
	}
	
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (K, NPQ)
	private MatrixBlock rotate180(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		this.input = input;
		int numRowsOutput = K;
		int numColsOutput = N * P * Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		i = 0; row = 0; 
		
		if(input.getNumColumns() != K*P*Q || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in reshape_col_rev:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		
		for (int k = 0; k < K; k++) {
			for (int n = 0; n < N; n++) {
				for (int p = 0; p < P; p++) {
					for (int q = 0; q < Q; q++) {		
						if(inputArray != null)
							outputArray[k*N*P*Q + n*P*Q + p*P + q] = inputArray[n*K*P*Q + k*P*Q + p*Q + q];
						else
							outputArray[k*N*P*Q + n*P*Q + p*P + q] = input.getValue(n, k*P*Q + p*Q + q);
					}
				}
			}
		}
		

		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for reshape_col_rev:" + execTime + " seconds.");
		return output;
	}
	
	// Reshape a matrix of dimension (K, NPQ) to 4D tensor of dimension (N, K, P, Q)
	private MatrixBlock reshape_col(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		
		this.input = input;
		int numRowsOutput = N;
		int numColsOutput = K * P * Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		i = 0; row = 0; 
		
		if(input.getNumColumns() != N*P*Q || input.getNumRows() != K) {
			throw new DMLRuntimeException("Incorrect input dimensions in reshape_col:" + input.getNumRows() + " " + input.getNumColumns());
		}
		
		
		for (int n = 0; n < N; n++) { 
			for (int k = 0; k < K; k++) { 
				for (int p = 0; p < P; p++) { 
					for (int q = 0; q < Q; q++) {
						if(inputArray != null)
							outputArray[n*K*P*Q + k*P*Q + p*Q + q] = inputArray[k*N*P*Q + n*P*Q + p*Q + q];
						else
							outputArray[n*K*P*Q + k*P*Q + p*Q + q] = input.getValue(k, n*P*Q + p*Q + q);
					}
				}
			}
		}
		

		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for reshape_col:" + execTime + " seconds.");
		return output;
	}
	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	private MatrixBlock col2im(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		
		this.input = input;
		MatrixBlock output = init(N, C * H * W);
		
		for (int c = 0; c < C; c++) { // Since format is NCHW
			for (int r = 0; r < R; r++) { // Get an input patch of size R X S
				for (int s = 0; s < S; s++) {
					for (int n = 0; n < N; n++) { // Do following for all images
						doCol2imOverInputPath_NCHW(n, c, r, s);
					}
				}
			}
		}
		
		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for col2im:" + execTime + " seconds.");
		return output;
	}
	
	// TODO:
	private void doCol2imOverInputPath_NCHW(int n, int c, int r, int s) {
		int input_row = r - pad_h;
		// And copy it to outputArray[i] (taking care of padding & striding)
		for (int p = P; p > 0; p--) {
			if (input_row >= 0 && input_row < H) {
				int input_col = s - pad_w;
				for (int q = Q; q > 0; q--, i++) {
					if (input_col >= 0 && input_col < W) {
						// Copy from [channel c, height input_row, width input_col]
						if (inputArray != null) {
							outputArray[n*C*H*W + c*H*W + input_row*W + input_col] += inputArray[i];
						}
						else {
							// TODO: Validate this
							// 4 X 4
							// 0 1 2  3
							// 4 5 6  7
							// 8 9 10 11
							int row = i / (N*P*Q);
							int col = i % (N*P*Q);
							outputArray[n*C*H*W + c*H*W + input_row*W + input_col] = input.getValue(row + 1, col + 1);
						}
					}
					input_col += stride_w;
				}
			} else {
				i += Q;
			}
			input_row += stride_h;
		}
	}
		

	// Converts a 4D tensor (N, C, R, S) to a matrix of dimension (CRS, NPQ)
	private MatrixBlock im2col(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		
		this.input = input;
		MatrixBlock output = init(C * R * S, N * P * Q);
		
		for (int c = 0; c < C; c++) { // Since format is NCHW
			for (int r = 0; r < R; r++) { // Get an input patch of size R X S
				for (int s = 0; s < S; s++) {
					for (int n = 0; n < N; n++) { // Do following for all images
						doIm2colOverInputPath_NCHW(n, c, r, s);
					}
				}
			}
		}
		
		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for im2col:" + execTime + " seconds.");
		return output;
	}
	
	private void doIm2colOverInputPath_NCHW(int n, int c, int r, int s) {
		int input_row = r - pad_h;
		// And copy it to outputArray[i] (taking care of padding & striding)
		for (int p = P; p > 0; p--) {
			if (input_row >= 0 && input_row < H) {
				int input_col = s - pad_w;
				for (int q = Q; q > 0; q--, i++) {
					if (input_col >= 0 && input_col < W) {
						// Copy from [channel c, height input_row, width input_col]
						if (inputArray != null)
							outputArray[i] = inputArray[n*C*H*W + c*H*W + input_row*W + input_col];
						else
							outputArray[i] = input.getValue(n + 1, c*H*W + input_row*W + input_col + 1);
					}
					input_col += stride_w;
				}
			} else {
				i += Q;
			}
			input_row += stride_h;
		}
	}

	
	
	private void checkInputDimensionForIm2col(MatrixBlock matBlock) throws DMLRuntimeException {
		if((N != matBlock.getNumRows() || C*H*W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
	}
	
	private void checkInputDimensionForCol2im(MatrixBlock matBlock) throws DMLRuntimeException {
		if((C*R*S != matBlock.getNumRows() || N*P*Q != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d_backward_data");
		}
	}
}

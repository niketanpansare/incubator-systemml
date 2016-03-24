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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.hops.OptimizerUtils;
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

	public static boolean ALLOW_MULTI_THREADED_OPS = false;
	public static boolean USE_IM2COL_POOLING = false; // TODO: Remove im2col-based pooling instruction
	
	private CPOperand _in2; // used for pooling backward
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
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
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
				|| opcode.equalsIgnoreCase("pooling_post_reshape")
				|| opcode.equalsIgnoreCase("maxpooling")) {
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
		} 
		else if (opcode.equalsIgnoreCase("pooling_backward_reshape")
				|| opcode.equalsIgnoreCase("maxpooling_backward")) {
			InstructionUtils.checkNumFields(parts, 15);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			out.split(parts[15]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[3]));
			stride.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));

			return new ConvolutionCPInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionCPInstruction: " + str);
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
				|| instOpcode.equalsIgnoreCase("pooling_post_reshape")
				|| instOpcode.equalsIgnoreCase("pooling_backward_reshape")
				|| instOpcode.equalsIgnoreCase("maxpooling")
				|| instOpcode.equalsIgnoreCase("maxpooling_backward")) {
			
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
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, C * R * S, N * P * Q);
				im2col(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("reshape_col")) {
				checkHeightWidth(ec);
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N, K * P * Q);
				reshape_col(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("rotate180")) {
				checkHeightWidth(ec);
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, K, N * P * Q);
				rotate180(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("col2im")) {
				checkHeightWidth(ec);
				checkInputDimensionForCol2im(matBlock);
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N, C * H * W);
				col2im(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("pooling_pre_reshape")) {
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N*C, H*W); 
				pooling_pre_reshape(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("pooling_post_reshape")) {
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N, C*P*Q);
				pooling_post_reshape(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("pooling_backward_reshape")) {
				MatrixBlock dout = ec.getMatrixInput(_in2.getName());
				this.input = dout;
				outputBlock = allocateOutputBlock(ec, C*R*S, N*P*Q, matBlock.getNonZeros());
				pooling_backward_reshape(matBlock, dout, outputBlock);
				ec.releaseMatrixInput(_in2.getName());
			}
			else if (instOpcode.equalsIgnoreCase("maxpooling")) {
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N, C*P*Q); 
				maxpooling(matBlock, outputBlock);
			}
			else if (instOpcode.equalsIgnoreCase("maxpooling_backward")) {
				MatrixBlock dout = ec.getMatrixInput(_in2.getName());
				this.input = matBlock;
				outputBlock = allocateDenseOutputBlock(ec, N, C*H*W); 
				maxpooling_backward(matBlock, dout, outputBlock);
				ec.releaseMatrixInput(_in2.getName());
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
	int globalIndex;  int row;

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
	
	private MatrixBlock allocateOutputBlock(ExecutionContext ec, int numRowsOutput, int numColsOutput, long nnz) throws DMLRuntimeException {
		MatrixBlock outputBlock = new MatrixBlock(numRowsOutput, numColsOutput, nnz);
		outputBlock.setNonZeros(nnz);
		return outputBlock;
	}
	
	private MatrixBlock allocateDenseOutputBlock(ExecutionContext ec, int numRowsOutput, int numColsOutput) throws DMLRuntimeException {
		MatrixBlock outputBlock = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		outputBlock.allocateDenseBlock();
		outputArray = outputBlock.getDenseBlock();
		outputBlock.setNonZeros(numRowsOutput * numColsOutput);
		return outputBlock;
	}

	private void maxpooling_backward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		if(input.getNumColumns() != C*H*W || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}

		double [] doutArray = null;
		if(!dout.isInSparseFormat()) 
			doutArray = dout.getDenseBlock();
		if(dout.getNumColumns() != C*P*Q || dout.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}

		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) {
				for (int c = 0; c < C; c++) {
					doPoolingBackward(c, n, doutArray, dout);
				}
			}
		}
		else {
			runParallelPoolingTask(constrainedNumThreads, false, doutArray, dout);
		}
	}

	public void doPoolingBackward(int c, int n, double [] doutArray, MatrixBlock dout) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				int start_index_h = p * stride_h - pad_h;
				int start_index_w = q * stride_w - pad_w;
				int end_index_h = Math.min(start_index_h + R, H);
				int end_index_w = Math.min(start_index_w + S, W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				//System.out.println("[" + start_index_h + " " + end_index_h + "]" + "[" + start_index_w + " " + end_index_w + "]");
				int maxIndex = n*C*H*W + c*H*W +  start_index_h*W + start_index_w; 
				double maxVal = Double.MIN_VALUE; 


				double currDoutVal = -1;
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						if(inputArray != null)
							currDoutVal = inputArray[n*C*H*W + c*H*W +  h*W + w];
						else
							currDoutVal = input.quickGetValue(n, c*H*W + h*W + w);

						if(maxVal < currDoutVal) {
							maxIndex = n*C*H*W + c*H*W +  h*W + w;
							maxVal = currDoutVal;
						}
					}
				}

				double inVal = -1;
				if(doutArray != null)
					inVal = doutArray[n*C*P*Q + c*P*Q +  p * Q + q];
				else
					inVal = dout.quickGetValue(n, c*P*Q +  p * Q + q);

				synchronized(this) {
					outputArray[maxIndex] += inVal;
				}
			}
		}
	}

	private void maxpooling(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(input.getNumColumns() != C*H*W || input.getNumRows() != N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int n = 0; n < N; n++) {
				for (int c = 0; c < C; c++) {
					doPooling(c, n);
				}
			}
		}
		else {
			runParallelPoolingTask(constrainedNumThreads, true, null, null);
		}	
	}

	public void doPooling(int c, int n) {
		for (int p = 0; p < P; p++) {
			for (int q = 0; q < Q; q++) {
				int start_index_h = p * stride_h - pad_h;
				int start_index_w = q * stride_w - pad_w;
				int end_index_h = Math.min(start_index_h + R, H);
				int end_index_w = Math.min(start_index_w + S, W);
				start_index_h = Math.max(start_index_h, 0);
				start_index_w = Math.max(start_index_w, 0);
				int out_index = n*C*P*Q + c*P*Q +  p * Q + q;
				//System.out.println("[" + start_index_h + " " + end_index_h + "]" + "[" + start_index_w + " " + end_index_w + "]");
				for (int h = start_index_h; h < end_index_h; h++) {
					for (int w = start_index_w; w < end_index_w; w++) {
						double inVal = -1;
						if(inputArray != null)
							inVal = inputArray[n*C*H*W + c*H*W +  h*W + w];
						else
							inVal = input.quickGetValue(n, c*H*W +  h*W + w);
						outputArray[out_index] = Math.max(outputArray[out_index], inVal);
					}
				}
			}
		}
	}

	private void runParallelPoolingTask(int constrainedNumThreads, boolean isForwardPooling, double [] doutArray, MatrixBlock dout) throws DMLRuntimeException {
		ArrayList<PoolingTask> tasks = new ArrayList<PoolingTask>();

		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c += TASK_SIZE) {
				tasks.add(new PoolingTask(this, n, c, isForwardPooling, doutArray, dout));
			}
		}

		ExecutorService pool = Executors.newFixedThreadPool( constrainedNumThreads );
		try {
			pool.invokeAll(tasks);
		} catch (InterruptedException e) {
			throw new DMLRuntimeException("Error while executing multi-threaded maxpooling", e);
		}	
		pool.shutdown();
	}

	private static class PoolingTask implements Callable<Object> {
		ConvolutionCPInstruction curr; int n1; int c1; boolean isForwardPooling;
		double [] doutArray; MatrixBlock dout;

		public PoolingTask(ConvolutionCPInstruction curr, int n, int c, boolean isForwardPooling, double [] doutArray, MatrixBlock dout) {
			this.curr = curr; 
			this.n1 = n;
			this.c1 = c;
			this.isForwardPooling = isForwardPooling;
			this.doutArray = doutArray;
			this.dout = dout;
		}

		@Override
		public Object call() throws Exception {
			for (int c = c1; c < Math.min(curr.C, c1 + TASK_SIZE); c++) {
				if(isForwardPooling)
					curr.doPooling(c, n1);
				else
					curr.doPoolingBackward(c, n1, doutArray, dout);
			}
			return null;
		}

	}
	
	// Using indices (matrix of dimension 1 X NCPQ) and values (tensor of shape [N, C, P, Q])
	// output a sparse matrix of dimension CRS X NPQ ... which is followed by col2im to output tensor of shape [N, C, H, W]
	// TODO: Fuse these two operations together for pooling.
	private void pooling_backward_reshape(MatrixBlock indices, MatrixBlock values, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		if(indices.getNumRows() != N*C*P*Q || indices.getNumColumns() != 1) {
			throw new DMLRuntimeException("Incorrect indices in pooling_backward_reshape");
		}
		if(values.getNumRows() != N || values.getNumColumns() != C*P*Q) {
			throw new DMLRuntimeException("Incorrect values in pooling_backward_reshape");
		}
		
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int p = 0; p < P; p++) {
					for (int q = 0; q < Q; q++) {
						// index will have range [1, pool_height*pool_width]
						int row_index = (int) (c*R*S + indices.quickGetValue(n*C*P*Q + c*P*Q + p*Q + q, 0)-1);
						double currentVal = outputBlock.quickGetValue(row_index, n*P*Q + p*Q + q);
						double updatedVal = currentVal + input.quickGetValue(n, c*P*Q + p*Q + q);
						outputBlock.setValue(row_index, n*P*Q + p*Q + q, updatedVal);
					}
				}
			}
		}
	}
	
	// Reshape a matrix of dimension (1, N*C*P*Q) of dimension a 4D tensor of dimension (N, C, P, Q)
	private void pooling_post_reshape(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		if(!input.isInSparseFormat()) {
			// TODO: Do in-place update
			double [] inputArray = input.getDenseBlock();
			for(int i = 0; i < inputArray.length; i++) {
				outputArray[i] = inputArray[i];
			}
			return;			
		}
		
		if(input.getNumColumns() != N*C*P*Q || input.getNumRows() != 1) {
			throw new DMLRuntimeException("Incorrect input dimensions in pooling_post_reshape:" + input.getNumRows() + " " + input.getNumColumns() + " " + N + " " + K*P*Q);
		}
		
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int p = 0; p < P; p++) {
					for (int q = 0; q < Q; q++) {
						outputArray[n*C*P*Q + c*P*Q + p*Q + q] = input.quickGetValue(0, n*C*P*Q + c*P*Q + p*Q + q);
					}
				}
			}
		}
	}

	// Reshape a 4D tensor of dimension (N, C, H, W) to a 4D tensor of dimension of dimension (N*C, 1, H, W)
	private void pooling_pre_reshape(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		
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
							outputArray[(n*C + c)*H*W + h*H + w] = input.quickGetValue(n, c*H*W + h*W + w);
					}
				}
			}
		}
	}
	
	// Reshape a 4D tensor of dimension (N, K, P, Q) to matrix of dimension (K, NPQ)
	private void rotate180(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		globalIndex = 0; row = 0; 
		
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
							outputArray[k*N*P*Q + n*P*Q + p*P + q] = input.quickGetValue(n, k*P*Q + p*Q + q);
					}
				}
			}
		}

	}
	
	// Reshape a matrix of dimension (K, NPQ) to 4D tensor of dimension (N, K, P, Q)
	private void reshape_col(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		globalIndex = 0; row = 0; 
		
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
							outputArray[n*K*P*Q + k*P*Q + p*Q + q] = input.quickGetValue(k, n*P*Q + p*Q + q);
					}
				}
			}
		}
		
	}
	
	// Converts a 4D tensor (N, C, R, S) to a matrix of dimension (CRS, NPQ)
	private void im2col(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		globalIndex = 0; row = 0;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			for (int c = 0; c < C; c++) { // Since format is NCHW
				for (int r = 0; r < R; r++) { // Get an input patch of size R X S
					for (int s = 0; s < S; s++) {
						for (int n = 0; n < N; n++) { // Do following for all images
							doIm2colOverInputPath_NCHW(n, c, r, s, this);
						}
					}
				}
			}
		}
		else {
			// Parallel im2col
			runParallelTask(constrainedNumThreads, true);
		}
		
	}
	
	public static final int TASK_SIZE = 64; // to take care of extremely small tasks 
	private void runParallelTask(int constrainedNumThreads, boolean isIm2Col) throws DMLRuntimeException {
		ArrayList<Im2ColOrCol2ImTask> tasks = new ArrayList<Im2ColOrCol2ImTask>();
		
		for (int c = 0; c < C; c++) { // Since format is NCHW
			for (int r = 0; r < R; r++) { // Get an input patch of size R X S
				for (int s = 0; s < S; s++) {
					for (int n = 0; n < N; n += TASK_SIZE) {
						tasks.add(new Im2ColOrCol2ImTask(this, c, r, s, n, isIm2Col));
					}
				}
			}
		}
		
		ExecutorService pool = Executors.newFixedThreadPool( constrainedNumThreads );
		try {
			pool.invokeAll(tasks);
		} catch (InterruptedException e) {
			throw new DMLRuntimeException("Error while executing multi-threaded im2col/col2im", e);
		}	
		pool.shutdown();
	}
	
	private static class Im2ColOrCol2ImTask implements Callable<Object> {
		ConvolutionCPInstruction curr; int c; int r; int s1; int n1; boolean isIm2Col;
		
		public Im2ColOrCol2ImTask(ConvolutionCPInstruction curr, int c, int r, int s, int n, boolean isIm2Col) {
			this.curr = curr; 
			this.c = c;
			this.r = r;
			this.s1 = s;
			this.n1 = n;
			this.isIm2Col = isIm2Col;
		}

		@Override
		public Object call() throws Exception {
			for (int n = n1; n < Math.min(curr.N, n1 + TASK_SIZE); n++) {
				if(isIm2Col)
					doIm2colOverInputPath_NCHW(n, c, r, s1, curr);
				else
					doCol2imOverInputPath_NCHW(n, c, r, s1, curr);
			}	
			return null;
		}
		
	}

	
	// Converts a matrix of dimension (CRS, NPQ) to a 4D tensor (N, C, H, W)
	private void col2im(MatrixBlock input, MatrixBlock outputBlock) throws DMLRuntimeException {
		inputArray = null;
		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		globalIndex = 0; row = 0;
		
		int constrainedNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(!ALLOW_MULTI_THREADED_OPS || constrainedNumThreads <= 1) {
			// Sequential col2im
			for (int c = 0; c < C; c++) { // Since format is NCHW
				for (int r = 0; r < R; r++) { // Get an input patch of size R X S
					for (int s = 0; s < S; s++) {
						for (int n = 0; n < N; n++) { // Do following for all images
							doCol2imOverInputPath_NCHW(n, c, r, s, this);
						}
					}
				}
			}
		}
		else {
			// Parallel col2im
			runParallelTask(constrainedNumThreads, false);
		}
	}
	
		
	private static void doCol2imOverInputPath_NCHW(int n, int c, int r, int s, ConvolutionCPInstruction inst) {
		int N = inst.N; int C = inst.C; int H = inst.H; int W = inst.W;
		int R = inst.R; int S = inst.S; int P = inst.P; int Q = inst.Q;
		int pad_h = inst.pad_h; int pad_w = inst.pad_w; int stride_h = inst.stride_h; int stride_w = inst.stride_w;
		int localIndex = ((c*R*S*N + r*S*N + s*N + n)*P*Q);
		
		int input_row = r - pad_h;
		// And copy it to outputArray[i] (taking care of padding & striding)
		for (int p = P; p > 0; p--) {
			if (input_row >= 0 && input_row < H) {
				int input_col = s - pad_w;
				for (int q = Q; q > 0; q--, localIndex++) {
					if (input_col >= 0 && input_col < W) {
						// Copy from [channel c, height input_row, width input_col]
						int index = n*C*H*W + c*H*W + input_row*W + input_col;
						if (inst.inputArray != null) {
							synchronized(inst) {
								inst.outputArray[index] += inst.inputArray[localIndex];
							}
						}
						else {
							// TODO: Validate this
							// 4 X 4
							// 0 1 2  3
							// 4 5 6  7
							// 8 9 10 11
							int row = localIndex / (N*P*Q);
							int col = localIndex % (N*P*Q);
							double val = inst.input.quickGetValue(row, col);
							synchronized(inst) {
								inst.outputArray[index] += val; 
							}
						}
					}
					input_col += stride_w;
				}
			} else {
				localIndex += Q;
			}
			input_row += stride_h;
		}
		
	}	
	
	private static void doIm2colOverInputPath_NCHW(int n, int c, int r, int s, ConvolutionCPInstruction inst) {
		int localIndex = ((c*inst.R*inst.S*inst.N + r*inst.S*inst.N + s*inst.N + n)*inst.P*inst.Q);
		
		int input_row = r - inst.pad_h;
		// And copy it to outputArray[i] (taking care of padding & striding)
		for (int p = inst.P; p > 0; p--) {
			if (input_row >= 0 && input_row < inst.H) {
				int input_col = s - inst.pad_w;
				for (int q = inst.Q; q > 0; q--, localIndex++) {
					if (input_col >= 0 && input_col < inst.W) {
						// Copy from [channel c, height input_row, width input_col]
						if (inst.inputArray != null)
							inst.outputArray[localIndex] = inst.inputArray[n*inst.C*inst.H*inst.W + c*inst.H*inst.W + input_row*inst.W + input_col];
						else
							inst.outputArray[localIndex] = inst.input.quickGetValue(n, c*inst.H*inst.W + input_row*inst.W + input_col);
					}
					input_col += inst.stride_w;
				}
			} else {
				localIndex += inst.Q;
			}
			input_row += inst.stride_h;
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

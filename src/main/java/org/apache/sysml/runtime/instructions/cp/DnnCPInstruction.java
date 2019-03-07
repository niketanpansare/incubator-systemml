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
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.DnnParameters;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN.PoolingType;
import org.apache.sysml.runtime.matrix.data.LibMatrixNative;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DnnUtils;
import org.apache.sysml.utils.NativeHelper;

public class DnnCPInstruction extends UnaryCPInstruction {
	private static final Log LOG = LogFactory.getLog(DnnCPInstruction.class.getName());
	private static boolean warnedUnderUtilitization = false;
	
	private final CPOperand _in2;
	private final CPOperand _in3;
	private final CPOperand _in4;
	private final CPOperand _in5;
	private final CPOperand _in6;
	private final CPOperand _in7;
	private final CPOperand _in8;
	private final CPOperand _out2;
	private final CPOperand _out3;
	private final CPOperand _out4;
	private final CPOperand _out5;
	private final ArrayList<CPOperand> _input_shape;
	private final ArrayList<CPOperand> _filter_shape;
	private final ArrayList<CPOperand> _stride;
	private final ArrayList<CPOperand> _padding;
	private final int _numThreads;
	private final double _intermediateMemoryBudget;
	
	public DnnCPInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, 
			ArrayList<CPOperand> stride, ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget, String opcode, String istr) {
		super(CPType.Dnn, null, in, out, opcode, istr);
		_in2 = in2;
		_in3 = in3;
		_in4 = null; _in5 = null; _in6 = null; _in7 = null; _in8 = null;
		_out2 = null; _out3 = null; _out4 = null; _out5 = null;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
		_numThreads = numThreads;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public DnnCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode, String istr, int numThreads, double intermediateMemoryBudget) {
		this(in, in2, null, out, null, null, null, null, numThreads, intermediateMemoryBudget, opcode, istr);
		if( !(opcode.equals("bias_add") || opcode.equals("relu_backward") || opcode.equals("bias_multiply") ) ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be bias_add or bias_multiply or relu_backward, but found " + opcode);
		}
	}
	
	private DnnCPInstruction(CPOperand in, CPOperand out, String opcode, String istr,
			ArrayList<CPOperand> stride, ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, null, null, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}
	
	public DnnCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, in2, null, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}
	
	public DnnCPInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, in2, in3, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}
	
	public DnnCPInstruction(CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand in5,
			CPOperand in6, CPOperand in7, CPOperand in8,
			CPOperand out, CPOperand out2, CPOperand out3, CPOperand out4, CPOperand out5, String opcode, String istr, 
			double intermediateMemoryBudget, int numThreads) throws DMLRuntimeException {
		super(CPType.Dnn, null, in1, out, opcode, istr);
		_in2 = in2;
		_in3 = in3;
		_in4 = in4;
		_in5 = in5;
		_in6 = in6;
		_in7 = in7;
		_in8 = in8;
		_out2 = out2;
		_out3 = out3;
		_out4 = out4;
		_out5 = out5;
		_stride = null;
		_padding = null;
		_input_shape = null;
		_filter_shape = null;
		_numThreads = numThreads;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}

	public static DnnCPInstruction parseInstruction(String str) {

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("maxpooling") || opcode.equalsIgnoreCase("relu_maxpooling") ||
			opcode.equalsIgnoreCase("avgpooling")) {
			InstructionUtils.checkNumFields(parts, 16);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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
			int k = Integer.parseInt(parts[15]);

			return new DnnCPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[16]));
		} 
		else if (opcode.equalsIgnoreCase("maxpooling_backward") || opcode.equalsIgnoreCase("relu_maxpooling_backward")
				|| opcode.equalsIgnoreCase("avgpooling_backward")
				|| opcode.equalsIgnoreCase("conv2d")
				|| opcode.equalsIgnoreCase("conv2d_backward_filter")
				|| opcode.equalsIgnoreCase("conv2d_backward_data")) {
			InstructionUtils.checkNumFields(parts, 17);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[15]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
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
			int k = Integer.parseInt(parts[16]);

			return new DnnCPInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[17]));
		}
		else if (opcode.equalsIgnoreCase("conv2d_bias_add")) {
			InstructionUtils.checkNumFields(parts, 18);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[16]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
			stride.add(new CPOperand(parts[4]));
			stride.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			padding.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			input_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));
			filter_shape.add(new CPOperand(parts[15]));
			int k = Integer.parseInt(parts[17]);

			return new DnnCPInstruction(in, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[18]));
		}
		else if (opcode.equalsIgnoreCase("bias_add") || opcode.equals("relu_backward") || opcode.equalsIgnoreCase("bias_multiply") ) {
			InstructionUtils.checkNumFields(parts, 5);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			int k = Integer.parseInt(parts[4]);
			return new DnnCPInstruction(in, in2, out, opcode, str, k, Double.parseDouble(parts[5]));
		}
		else if (opcode.equalsIgnoreCase("batch_norm2d")) {
			InstructionUtils.checkNumFields(parts, 13);
			CPOperand in1 = new CPOperand(parts[1]); // image
			CPOperand in2 = new CPOperand(parts[2]); // scale
			CPOperand in3 = new CPOperand(parts[3]); // bias
			CPOperand in4 = new CPOperand(parts[4]); // runningMean
			CPOperand in5 = new CPOperand(parts[5]); // runningVar
			CPOperand in6 = new CPOperand(parts[6]); // mode
			CPOperand in7 = new CPOperand(parts[7]); // epsilon
			CPOperand in8 = new CPOperand(parts[8]); // exponentialAverageFactor
			CPOperand out = new CPOperand(parts[9]);  // ret
			CPOperand out2 = new CPOperand(parts[10]); // retRunningMean
			CPOperand out3 = new CPOperand(parts[11]); // retRunningVar
			CPOperand out4 = new CPOperand(parts[12]); // resultSaveMean
			CPOperand out5 = new CPOperand(parts[13]); // resultSaveInvVariance
			return new DnnCPInstruction(in1, in2, in3, in4, in5, in6, in7, in8, out, out2, out3, out4, out5, opcode, str, 0, 0);
		}
		else if (opcode.equalsIgnoreCase("batch_norm2d_backward")) {
			InstructionUtils.checkNumFields(parts, 9);
			CPOperand in1 = new CPOperand(parts[1]); // image
			CPOperand in2 = new CPOperand(parts[2]); // dout
			CPOperand in3 = new CPOperand(parts[3]); // scale
			CPOperand in4 = new CPOperand(parts[4]); // epsilon
			CPOperand in5 = new CPOperand(parts[5]); // resultSaveMean
			CPOperand in6 = new CPOperand(parts[6]); // resultSaveInvVariance
			CPOperand out = new CPOperand(parts[7]);  // dX
			CPOperand out2 = new CPOperand(parts[8]); // dScale
			CPOperand out3 = new CPOperand(parts[9]); // dBias
			return new DnnCPInstruction(in1, in2, in3, in4, in5, in6, null, null, out, out2, out3, null, null, opcode, str, 0, 0);
		}
		else if (opcode.equalsIgnoreCase("lstm")) {
			InstructionUtils.checkNumFields(parts, 9);
			CPOperand in1 = new CPOperand(parts[1]); // X
			CPOperand in2 = new CPOperand(parts[2]); // W
			CPOperand in3 = new CPOperand(parts[3]); // b
			CPOperand in4 = new CPOperand(parts[4]); // out0
			CPOperand in5 = new CPOperand(parts[5]); // c0
			CPOperand in6 = new CPOperand(parts[6]); // return_seq
			CPOperand out = new CPOperand(parts[7]);  // out
			CPOperand out2 = new CPOperand(parts[8]); // c
			int numThreads = Integer.parseInt(parts[9]);
			return new DnnCPInstruction(in1, in2, in3, in4, in5, in6, null, null, out, out2, null, null, null, opcode, str, 0, numThreads);
		}
		else if (opcode.equalsIgnoreCase("lstm_backward")) {
			InstructionUtils.checkNumFields(parts, 14);
			CPOperand in1 = new CPOperand(parts[1]); // X
			CPOperand in2 = new CPOperand(parts[2]); // W
			CPOperand in3 = new CPOperand(parts[3]); // b
			CPOperand in4 = new CPOperand(parts[4]); // out0
			CPOperand in5 = new CPOperand(parts[5]); // c0
			CPOperand in6 = new CPOperand(parts[6]); // return_seq
			CPOperand in7 = new CPOperand(parts[7]); // dout
			CPOperand in8 = new CPOperand(parts[8]); // dc
			CPOperand out = new CPOperand(parts[9]);  // dX
			CPOperand out2 = new CPOperand(parts[10]); // dW
			CPOperand out3 = new CPOperand(parts[11]); // db
			CPOperand out4 = new CPOperand(parts[12]); // dout0
			CPOperand out5 = new CPOperand(parts[13]); // dc0
			int numThreads = Integer.parseInt(parts[14]);
			return new DnnCPInstruction(in1, in2, in3, in4, in5, in6, in7, in8, out, out2, out3, out4, out5, opcode, str, 0, numThreads);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a DnnCPInstruction: " + str);
		}
	}

	private static int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL, int index) {
		return (int) ec.getScalarInput(aL.get(index).getName(),
			aL.get(index).getValueType(), aL.get(index).isLiteral()).getLongValue();
	}
	
	private String getPrefixForTempCacheVar(ExecutionContext ec) {
		if(instOpcode.equalsIgnoreCase("lstm") || instOpcode.equalsIgnoreCase("lstm_backward")) {
			return "___cache_" + ec.getMatrixObject(input1.getName()).getUniqueIdVersion() 
					+ "_" + ec.getMatrixObject(_in2.getName()).getUniqueIdVersion()
					+ "_" + ec.getMatrixObject(_in3.getName()).getUniqueIdVersion()
					+ "_" + ec.getMatrixObject(_in4.getName()).getUniqueIdVersion()
					+ "_" + ec.getMatrixObject(_in5.getName()).getUniqueIdVersion()
					+ "_" + ec.getScalarInput(_in6.getName(), _in6.getValueType(), _in6.isLiteral());
		}
		else {
			throw new DMLRuntimeException("The instruction " + instOpcode + " is not eligible for temporary cache variable");
		}
	}
	private String getScopeVarForTempCacheVar() {
		if(instOpcode.equalsIgnoreCase("lstm") || instOpcode.equalsIgnoreCase("lstm_backward")) {
			return input1.getName();
		}
		else {
			throw new DMLRuntimeException("The instruction " + instOpcode + " is not eligible for temporary cache variable");
		}
	}
	
	public void processLstmInstruction(ExecutionContext ec) {
		MatrixBlock X = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock W = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock b = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
		MatrixBlock out0 = ec.getMatrixInput(_in4.getName(), getExtendedOpcode());
		MatrixBlock c0 = ec.getMatrixInput(_in5.getName(), getExtendedOpcode());
		boolean return_seq = ec.getScalarInput(_in6.getName(), _in6.getValueType(), _in6.isLiteral()).getBooleanValue();
		
		int N = X.getNumRows();
		int TD = X.getNumColumns();
		int DPlusM = W.getNumRows();
		int M = W.getNumColumns() / 4;
		if(b.getNumRows() != 1 || b.getNumColumns() != M*4) {
			throw new DMLRuntimeException("Incorrect dimensions of bias in lstm instruction. Expected [1, " + (M*4) + "], "
					+ "but found [" + b.getNumRows() + "," + b.getNumColumns() + "]");
		}
		if(out0.getNumRows() != N || out0.getNumColumns() != M) {
			throw new DMLRuntimeException("Incorrect dimensions of out0 in lstm instruction. Expected [" + N + ", " + M + "], "
					+ "but found [" + out0.getNumRows() + "," + out0.getNumColumns() + "]");
		}
		if(c0.getNumRows() != N || c0.getNumColumns() != M) {
			throw new DMLRuntimeException("Incorrect dimensions of c0 in lstm instruction. Expected [" + N + ", " + M + "], "
					+ "but found [" + out0.getNumRows() + "," + out0.getNumColumns() + "]");
		}
		int D = DPlusM - M;
		int T = TD / D;
		
		MatrixBlock out = new MatrixBlock(N, return_seq ? (T*M) : M, false);
		MatrixBlock c = new MatrixBlock(N, M, false);
		
		if(ConfigurationManager.allocateNNCache()) {
			MatrixBlock cache_out = new MatrixBlock(T, N*M, false);
			MatrixBlock cache_c = new MatrixBlock(T, N*M, false);
			MatrixBlock cache_ifog = new MatrixBlock(T, N*4*M, false);
			cache_out.allocateDenseBlock();
			cache_c.allocateDenseBlock();
			cache_ifog.allocateDenseBlock();
			LibMatrixDNN.lstm(X, W, b, out0, c0, 
					return_seq, N, T, D, M,
					out,  c, cache_out, cache_c, cache_ifog,
					_numThreads);
			String prefixTempCache = getPrefixForTempCacheVar(ec);
			ec.setTemporaryCacheMatrix(prefixTempCache + "_cp_cache_out", cache_out, getScopeVarForTempCacheVar());
			ec.setTemporaryCacheMatrix(prefixTempCache + "_cp_cache_c", cache_c, getScopeVarForTempCacheVar());
			ec.setTemporaryCacheMatrix(prefixTempCache + "_cp_cache_ifog", cache_ifog, getScopeVarForTempCacheVar());
		}
		else {
			LibMatrixDNN.lstm(X, W, b, out0, c0, 
					return_seq, N, T, D, M,
					out,  c, null, null, null,
					_numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in4.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in5.getName(), getExtendedOpcode());
		ec.setMatrixOutput(output.getName(), out, getExtendedOpcode());
		ec.setMatrixOutput(_out2.getName(), c, getExtendedOpcode());
	}
	
	public void processLstmBackwardInstruction(ExecutionContext ec) {
		MatrixBlock X = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock W = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock b = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
		MatrixBlock out0 = ec.getMatrixInput(_in4.getName(), getExtendedOpcode());
		MatrixBlock c0 = ec.getMatrixInput(_in5.getName(), getExtendedOpcode());
		boolean return_seq = ec.getScalarInput(_in6.getName(), _in6.getValueType(), _in6.isLiteral()).getBooleanValue();
		MatrixBlock dout = ec.getMatrixInput(_in7.getName(), getExtendedOpcode());
		MatrixBlock dc = ec.getMatrixInput(_in8.getName(), getExtendedOpcode());
		
		int N = X.getNumRows();
		int TD = X.getNumColumns();
		int DPlusM = W.getNumRows();
		int M = W.getNumColumns() / 4;
		int D = DPlusM - M;
		int T = TD / D;
		if(b.getNumRows() != 1 || b.getNumColumns() != M*4) {
			throw new DMLRuntimeException("Incorrect dimensions of bias in lstm_backward instruction. Expected [1, " + (M*4) + "], "
					+ "but found [" + b.getNumRows() + "," + b.getNumColumns() + "]");
		}
		if(out0.getNumRows() != N) {
			throw new DMLRuntimeException("Unsupported operation: The batch size of previous iteration " + out0.getNumRows() + 
					" is different than the batch size of current iteration " + N);
		}
		if(out0.getNumColumns() != M) {
			throw new DMLRuntimeException("Incorrect dimensions of out0 in lstm_backward instruction. Expected [" + N + ", " + M + "], "
					+ "but found [" + out0.getNumRows() + "," + out0.getNumColumns() + "]");
		}
		if(c0.getNumRows() != N || c0.getNumColumns() != M) {
			throw new DMLRuntimeException("Incorrect dimensions of c0 in lstm_backward instruction. Expected [" + N + ", " + M + "], "
					+ "but found [" + out0.getNumRows() + "," + out0.getNumColumns() + "]");
		}
		if(dout.getNumRows() != N || dout.getNumColumns() != (return_seq ? (T*M) : M)) {
			throw new DMLRuntimeException("Incorrect dimensions of dout in lstm_backward instruction. Expected [" + N + ", " + (return_seq ? (T*M) : M) + "], "
					+ "but found [" + dout.getNumRows() + "," + dout.getNumColumns() + "]");
		}
		if(dc.getNumRows() != N || dc.getNumColumns() != M) {
			throw new DMLRuntimeException("Incorrect dimensions of dc in lstm_backward instruction. Expected [" + N + ", " + M + "], "
					+ "but found [" + dc.getNumRows() + "," + dc.getNumColumns() + "]");
		}
		
		MatrixBlock cache_out = null;
		MatrixBlock cache_c = null;
		MatrixBlock cache_ifog = null;
		if(ConfigurationManager.allocateNNCache()) {
			String prefixTempCache = getPrefixForTempCacheVar(ec);
			if(ec.containsTemporaryCacheMatrix(prefixTempCache + "_cp_cache_out", getScopeVarForTempCacheVar()) && 
					ec.containsTemporaryCacheMatrix(prefixTempCache + "_cp_cache_c", getScopeVarForTempCacheVar()) &&
					ec.containsTemporaryCacheMatrix(prefixTempCache + "_cp_cache_ifog", getScopeVarForTempCacheVar())) {
				cache_out = ec.getMatrixInput(prefixTempCache + "_cp_cache_out", getExtendedOpcode());
				cache_c = ec.getMatrixInput(prefixTempCache + "_cp_cache_c", getExtendedOpcode());
				cache_ifog = ec.getMatrixInput(prefixTempCache + "_cp_cache_ifog", getExtendedOpcode());
			}
			else {
				System.out.print("processLstmBackwardInstruction: [");
				for(Entry<String, Data> kv : ec.getVariables().entrySet()) {
					if(kv.getValue() instanceof MatrixObject) {
						System.out.print(" " + kv.getKey() + "->" + 
								((MatrixObject)kv.getValue()).getUniqueIdVersion() + "{");
						for(String cacheData : ((MatrixObject)kv.getValue()).getTemporaryCacheData()) {
							System.out.print(" " + cacheData);
						}
						System.out.print("}");
					}
				}
				System.out.println("]");
				// Only warn when ConfigurationManager.allocateNNCache() is true
				ArrayList<String> varList = ec.getVarList();
				if(varList.contains(prefixTempCache + "_cp_cache_out") && 
						varList.contains(prefixTempCache + "_cp_cache_c") &&
						varList.contains(prefixTempCache + "_cp_cache_ifog")) {
					LOG.warn("Invoking lstm forward function redundantly in lstm_backward call. "
							+ "Note: the cache variables are present in the execution context but not associated with the scope variables."
							+ " This can sometime happen due to cpvar/mvvar");
				}
				else {
					LOG.warn("Invoking lstm forward function redundantly in lstm_backward call.");
				}
			}
		}
		
		if(cache_out == null) {
			cache_out = new MatrixBlock(T, N*M, false);
			cache_c = new MatrixBlock(T, N*M, false);
			cache_ifog = new MatrixBlock(T, N*4*M, false);
			cache_out.allocateDenseBlock();
			cache_c.allocateDenseBlock();
			cache_ifog.allocateDenseBlock();
			LibMatrixDNN.lstm(X, W, b, out0, c0, 
					return_seq, N, T, D, M,
					// Avoid out and c computation in lstm forward call
					null, null, 
					cache_out, cache_c, cache_ifog,
					_numThreads);
		}
		
		MatrixBlock dX = new MatrixBlock(N, T*D, false);
		MatrixBlock dW = new MatrixBlock(D+M, 4*M, false);
		MatrixBlock db = new MatrixBlock(1, 4*M, false);
		MatrixBlock dout0 = new MatrixBlock(N, M, false);
		MatrixBlock dc0 = new MatrixBlock(N, M, false);
		LibMatrixDNN.lstm_backward(dout, dc, X, W, b, out0, c0, return_seq, N, T, D, M,
				cache_out, cache_c, cache_ifog, // from forward invocation
				dX, dW, db, dout0, dc0, // output
				_numThreads);
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in4.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in5.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in7.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in8.getName(), getExtendedOpcode());
		ec.setMatrixOutput(output.getName(), dX, getExtendedOpcode());
		ec.setMatrixOutput(_out2.getName(), dW, getExtendedOpcode());
		ec.setMatrixOutput(_out3.getName(), db, getExtendedOpcode());
		ec.setMatrixOutput(_out4.getName(), dout0, getExtendedOpcode());
		ec.setMatrixOutput(_out5.getName(), dc0, getExtendedOpcode());
	}
	
	public void processReluBackwardInstruction(ExecutionContext ec) {
		// (X > 0) * dout
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(),
			input.isInSparseFormat() || dout.isInSparseFormat() );
		
		if( !input.isEmpty() && !dout.isEmpty() ) { //sparse-safe
			outputBlock.allocateBlock();
			LibMatrixDNN.reluBackward(input, dout, outputBlock, _numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processBiasAddInstruction(ExecutionContext ec) {
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock = null;
		
		if(bias.getNumColumns() != 1) {
			throw new DMLRuntimeException("Expected the number of columns of bias matrix to be 1, but found " + bias.getNumColumns());
		}
		
		if(input.isEmpty() && bias.isEmpty()) {
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), true);
		}
		else if(bias.isEmpty()) {
			outputBlock = new MatrixBlock(input);
		}
		else {
			// As we always fill the output first with bias
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), false);
			outputBlock.allocateDenseBlock();
			LibMatrixDNN.biasAdd(input, bias, outputBlock, _numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processBiasMultiplyInstruction(ExecutionContext ec) {
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock = null;
		
		if(bias.getNumColumns() != 1) {
			throw new DMLRuntimeException("Expected the number of columns of bias matrix to be 1, but found " + bias.getNumColumns());
		}
		
		if(bias.isEmpty()) {
			// Anything multiplied by zero is zero
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), true);
		}
		else {
			// As we always fill the output first with bias
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), 
				input.isInSparseFormat()).allocateBlock();
			LibMatrixDNN.biasMultiply(input, bias, outputBlock, _numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processBatchNorm2dInstruction(ExecutionContext ec) {
		MatrixBlock image = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock scale = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock bias = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
		MatrixBlock runningMean = ec.getMatrixInput(_in4.getName(), getExtendedOpcode());
		MatrixBlock runningVar = ec.getMatrixInput(_in5.getName(), getExtendedOpcode());
		String phase = ec.getScalarInput(_in6.getName(), _in6.getValueType(), _in6.isLiteral()).getStringValue();
		double epsilon = ec.getScalarInput(_in7.getName(), _in7.getValueType(), _in7.isLiteral()).getDoubleValue();
		double mu = ec.getScalarInput(_in8.getName(), _in8.getValueType(), _in8.isLiteral()).getDoubleValue();
		
		MatrixBlock ret = new MatrixBlock(image.getNumRows(), image.getNumColumns(), false).allocateBlock();
		MatrixBlock retRunningMean = new MatrixBlock(runningMean.getNumRows(), runningMean.getNumColumns(), false).allocateBlock();
		MatrixBlock retRunningVar = new MatrixBlock(runningVar.getNumRows(), runningVar.getNumColumns(), false).allocateBlock();
		MatrixBlock resultSaveMean = new MatrixBlock(runningMean.getNumRows(), runningMean.getNumColumns(), false).allocateBlock();
		MatrixBlock resultSaveInvVariance = new MatrixBlock(runningVar.getNumRows(), runningVar.getNumColumns(), false).allocateBlock();
		
		LibMatrixDNN.batchNorm2D(image, scale, bias, runningMean, runningVar, phase, epsilon, mu, ret, 
				retRunningMean, retRunningVar, resultSaveMean, resultSaveInvVariance);
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in4.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in5.getName(), getExtendedOpcode());
		ec.setMatrixOutput(output.getName(), ret, getExtendedOpcode());
		ec.setMatrixOutput(_out2.getName(), retRunningMean, getExtendedOpcode());
		ec.setMatrixOutput(_out3.getName(), retRunningVar, getExtendedOpcode());
		ec.setMatrixOutput(_out4.getName(), resultSaveMean, getExtendedOpcode());
		ec.setMatrixOutput(_out5.getName(), resultSaveInvVariance, getExtendedOpcode());
	}
	
	public void processBatchNorm2dBackwardInstruction(ExecutionContext ec) {
		MatrixBlock image = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock scale = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
		double epsilon = ec.getScalarInput(_in4.getName(), _in4.getValueType(), _in4.isLiteral()).getDoubleValue();
		MatrixBlock resultSaveMean = ec.getMatrixInput(_in5.getName(), getExtendedOpcode());
		MatrixBlock resultSaveInvVariance = ec.getMatrixInput(_in6.getName(), getExtendedOpcode());
		
		MatrixBlock dX = new MatrixBlock(image.getNumRows(), image.getNumColumns(), false).allocateBlock();
		MatrixBlock dScale = new MatrixBlock(scale.getNumRows(), scale.getNumColumns(), false).allocateBlock();
		MatrixBlock dBias = new MatrixBlock(scale.getNumRows(), scale.getNumColumns(), false).allocateBlock();
		
		LibMatrixDNN.batchNorm2DBackward(image, dout, scale, epsilon, resultSaveMean, resultSaveInvVariance, dX, dScale, dBias);
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in5.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in6.getName(), getExtendedOpcode());
		ec.setMatrixOutput(output.getName(), dX, getExtendedOpcode());
		ec.setMatrixOutput(_out2.getName(), dScale, getExtendedOpcode());
		ec.setMatrixOutput(_out3.getName(), dBias, getExtendedOpcode());
	}
	
	
	// Assumption: enableNative && NativeHelper.isNativeLibraryLoaded() is true
	// This increases the number of native calls. For example:the cases where filter is sparse but input is dense
	private static boolean isFilterSparse(MatrixBlock filter) {
		long numElems = filter.getNumRows()*filter.getNumColumns();
		// if filter is less than 10 MB in dense format (which handles almost all the cases).
		// In fact, using threshold of 1 MB is still sufficient for common CNNs.
		if(filter.isInSparseFormat() && numElems < 10e+6)
			filter.sparseToDense(); 
		return filter.isInSparseFormat();
	}
	
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		
		if (instOpcode.equalsIgnoreCase("bias_add")) {
			processBiasAddInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("bias_multiply")) {
			processBiasMultiplyInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("relu_backward")) {
			processReluBackwardInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("batch_norm2d")) {
			processBatchNorm2dInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("batch_norm2d_backward")) {
			processBatchNorm2dBackwardInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("lstm")) {
			processLstmInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("lstm_backward")) {
			processLstmBackwardInstruction(ec);
			return;
		}
		
		// acquire inputs
		MatrixBlock outputBlock = null;
		MatrixBlock matBlock = instOpcode.equalsIgnoreCase("avgpooling_backward") ? null : ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		int pad_h = getScalarInput(ec, _padding, 0);
		int pad_w = getScalarInput(ec, _padding, 1);
		int stride_h = getScalarInput(ec, _stride, 0);
		int stride_w = getScalarInput(ec, _stride, 1);

		int N = getScalarInput(ec, _input_shape, 0);
		int C = getScalarInput(ec, _input_shape, 1);
		int H = getScalarInput(ec, _input_shape, 2);
		int W = getScalarInput(ec, _input_shape, 3);

		int K = getScalarInput(ec, _filter_shape, 0);
		
		int R = getScalarInput(ec, _filter_shape, 2);
		int S = getScalarInput(ec, _filter_shape, 3);
		int P = (int) DnnUtils.getP(H, R, stride_h, pad_h);
		int Q = (int) DnnUtils.getQ(W, S, stride_w, pad_w);
		
		DnnParameters params = new DnnParameters(N, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w, _numThreads);
		params.enableNative = NativeHelper.isNativeLibraryLoaded();
		if (instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling") ||
			instOpcode.equalsIgnoreCase("avgpooling")) {
			if(matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, C*P*Q, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C*P*Q, false).allocateBlock();
				
				PoolingType poolType = (instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling")) ? PoolingType.MAX : PoolingType.AVG;
				if(instOpcode.equalsIgnoreCase("relu_maxpooling"))
					params.minValForMaxPoolOperations = 0;
				LibMatrixDNN.pooling(matBlock, outputBlock, params, poolType);
			}
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward") || instOpcode.equalsIgnoreCase("relu_maxpooling_backward") ||
				instOpcode.equalsIgnoreCase("avgpooling_backward")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			boolean isEmpty = instOpcode.equalsIgnoreCase("avgpooling_backward") ? dout.isEmpty() : (matBlock.isEmpty() || dout.isEmpty());
			if(isEmpty) {
				outputBlock = new MatrixBlock(N, C*H*W, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C*H*W, false).allocateBlock();
				PoolingType poolType = (instOpcode.equalsIgnoreCase("maxpooling_backward") || instOpcode.equalsIgnoreCase("relu_maxpooling_backward")) ? PoolingType.MAX : PoolingType.AVG;
				boolean performReLUBackward = instOpcode.equalsIgnoreCase("relu_maxpooling_backward");
				if(performReLUBackward)
					params.minValForMaxPoolOperations = 0;
				LibMatrixDNN.poolingBackward(matBlock, dout, outputBlock, params, performReLUBackward, poolType);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d")) {
			resetNumThreads(params, C*R*S, P*Q, matBlock.getNonZeros() / (matBlock.getNumRows()*matBlock.getNumColumns()));
			MatrixBlock filter = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(filter.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, K*P*Q, true);
			}
			else {
				boolean sparse = matBlock.isUltraSparse(false) && params.bias == null
					&& matBlock.getInMemorySize() < MatrixBlock.estimateSizeDenseInMemory(N, K*P*Q);
				outputBlock = new MatrixBlock(N, K*P*Q, sparse).allocateBlock();
				if(params.enableNative && !isFilterSparse(filter) && !matBlock.isInSparseFormat())
					LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
				else
					LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
			resetNumThreads(params, C*R*S, P*Q, matBlock.getNonZeros() / (matBlock.getNumRows()*matBlock.getNumColumns()));
			MatrixBlock filter = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
			MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(bias.getNumRows() != params.K || bias.getNumColumns() != 1) {
				throw new DMLRuntimeException("Incorrect shape of bias matrix: [" + bias.getNumRows() + " " + bias.getNumColumns() + "]. "
						+ "Expected: [" + params.K + ", 1]");
			}
			boolean isOutputConvEmpty = filter.isEmpty() || matBlock.isEmpty();
			if(isOutputConvEmpty && bias.isEmpty()) {
				// bias_add(empty mb, empty mb) = empty mb
				outputBlock = new MatrixBlock(N, K*P*Q, true);
			}
			else if(isOutputConvEmpty && !bias.isEmpty()) {
				// Add bias to empty output block
				// bias_add(empty mb, bias)
				outputBlock = new MatrixBlock(N, K*P*Q, false).allocateBlock();
				for(int n = 0;  n < params.N; n++) 
					DnnUtils.fillBias(bias, outputBlock.getDenseBlockValues(),
						n, n+1, params.N, params.K, params.P*params.Q);
			}
			else {
				outputBlock = new MatrixBlock(N, K*P*Q, false).allocateBlock();
				if(!bias.isEmpty()) {
					// Handle situation where both input and filter are non empty, but bias is empty
					params.bias = bias;
				}
				if(params.enableNative && !isFilterSparse(filter) && !matBlock.isInSparseFormat())
					LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
				else
					LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
			}
			ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_filter")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(dout.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(K, C*R*S, true);
			}
			else {
				outputBlock = new MatrixBlock(K, C*R*S, false).allocateBlock();
				if(params.enableNative && !matBlock.isInSparseFormat() && !dout.isInSparseFormat())
					LibMatrixNative.conv2dBackwardFilter(matBlock, dout, outputBlock, params);
				else
					LibMatrixDNN.conv2dBackwardFilter(matBlock, dout, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_data")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(dout.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, C * H * W, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C * H * W, false).allocateBlock();
				if(params.enableNative && !isFilterSparse(matBlock) && !dout.isInSparseFormat())
					LibMatrixNative.conv2dBackwardData(matBlock, dout, outputBlock, params);
				else
					LibMatrixDNN.conv2dBackwardData(matBlock, dout, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		
		// release inputs/outputs
		if(!instOpcode.equalsIgnoreCase("avgpooling_backward"))
			ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	/**
	 * Reset the number of thread to respect the intermediate CP memory budget
	 * 
	 * @param params convolution parameters
	 * @param numRows number of rows of intermediate matrix used per thread
	 * @param numCols number of rows of intermediate matrix used per thread
	 * @param sparsity sparsity of intermediate matrix used per thread
	 */
	private void resetNumThreads(DnnParameters params, int numRows, int numCols, double sparsity) {
		if(ConfigurationManager.isGPU()) {
			double memBudget1Thread = OptimizerUtils.estimateSizeExactSparsity(numRows, numCols, sparsity);
			int limitedDegreeOfParallelism = (int) Math.floor(_intermediateMemoryBudget / memBudget1Thread);
			if(params.numThreads > limitedDegreeOfParallelism) {
				params.numThreads = limitedDegreeOfParallelism;
				if(!warnedUnderUtilitization)
					LOG.warn("CPU Under-utilization to respect the intermediate memory budget. To avoid this, please try reducing the mini-batch or forcing gpu execution.");
				warnedUnderUtilitization = true;
			}
		}
	}
}

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

package org.apache.sysml.runtime.instructions.gpu;

import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.lops.DataGen;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.utils.GPUStatistics;

public class DataGenGPUInstruction extends GPUInstruction {
	
	private DataGenMethod method = DataGenMethod.INVALID;
	private CPOperand input1 = null;
	private CPOperand output = null;

	private final CPOperand rows, cols;
	private final double minValue, maxValue, sparsity;
	private final String pdfStr;
	private final long seed;
	
	private DataGenGPUInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, double minValue, double maxValue, double sparsity, long seed,
			String probabilityDensityFunction, String pdfParams, String opcode, String istr) {
		super(op, opcode, istr);
		this.method = mthd;
		this.input1 = in;
		this.output = out;
		this.rows = rows;
		this.cols = cols;
//		this.rowsInBlock = rpb;
//		this.colsInBlock = cpb;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.pdfStr = probabilityDensityFunction;
//		this.pdfParams = pdfParams;
	}
	
	public static DataGenGPUInstruction parseInstruction(String str) throws DMLRuntimeException {
		DataGenMethod method = DataGenMethod.INVALID;

		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = s[0];
		
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( s, 11 );
		}
		else  {
			throw new DMLRuntimeException("Unsupported opcode:" + opcode); 
		}
		
		CPOperand out = new CPOperand(s[s.length-1]);
		Operator op = null;
		
		if ( method == DataGenMethod.RAND ) 
		{
			CPOperand rows = new CPOperand(s[1]);
			CPOperand cols = new CPOperand(s[2]);
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			double minValue = !s[5].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[5]).doubleValue() : -1;
			double maxValue = !s[6].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[6]).doubleValue() : -1;
			double sparsity = !s[7].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[7]).doubleValue() : -1;
			long seed = !s[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Long.valueOf(s[8]).longValue() : -1;
			String pdf = s[9];
			String pdfParams = !s[10].contains( Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[10] : null;
			// int k = Integer.parseInt(s[11]);
			
			return new DataGenGPUInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, pdfParams, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		GPUStatistics.incrementNoOfExecutedGPUInst();

		if ( method == DataGenMethod.RAND ) {
			long lrows = ec.getScalarInput(rows).getLongValue();
			long lcols = ec.getScalarInput(cols).getLongValue();
			
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
				lSeed = DataGenOp.generateRandomSeed();
			
			MatrixObject in1 = getMatrixInputForGPUInstruction(ec, input1.getName());
			ec.setMetaData(output.getName(), lrows, lcols);
			RandomMatrixGenerator.PDF pdf = RandomMatrixGenerator.PDF.valueOf(pdfStr.toUpperCase());
			switch (pdf) {
				case UNIFORM:
					LibMatrixCUDA.randomUniform(ec, ec.getGPUContext(0), getExtendedOpcode(), in1, output.getName(), lrows, lcols, minValue, maxValue, sparsity, lSeed);
					break;
				default:
					throw new DMLRuntimeException("Unsupported pdf:" + pdfStr); 
			}
			ec.releaseMatrixInputForGPUInstruction(input1.getName());
			ec.releaseMatrixOutputForGPUInstruction(output.getName());
		}
		else {
			throw new DMLRuntimeException("Unsupported method:" + method.name());
		}
	    
	}

}

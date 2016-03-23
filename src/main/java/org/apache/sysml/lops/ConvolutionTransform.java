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

package org.apache.sysml.lops;

import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

public class ConvolutionTransform extends Lop
{

	
	public enum OperationTypes {
		IM2COL,
		RESHAPE_COL,
		ROTATE180,
		COL2IM,
		POOLING_PRE_RESHAPE, POOLING_POST_RESHAPE,
		POOLING_BACKWARD_RESHAPE,
		MAX_POOLING
	};
	
	private OperationTypes operation = null;
	
	/**
	 * Constructor when we have one input.
	 * @param input
	 * @param op
	 */

	public ConvolutionTransform(Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, et);
	}
	
	public ConvolutionTransform(Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Transform, dt, vt);		
		init(input, op, dt, vt, ExecType.MR);
	}
	
	private void init (Lop input, ConvolutionTransform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		operation = op;
 
		this.addInput(input);
		input.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
			throw new RuntimeException("The execution type is not supported: " + et.name());
		}
		else //CP/SPARK
		{
			// <code>breaksAlignment</code> is not meaningful when <code>Transform</code> executes in CP. 
			breaksAlignment = false;
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}

	@Override
	public String toString() {

		return " Operation: " + operation;
	}

	/**
	 * method to get operation type
	 * @return
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}

	private String getOpcode() {
		switch(operation) {
			
		case IM2COL:
			return "im2col";
			
		case RESHAPE_COL:
			return "reshape_col";
		
		case ROTATE180:
			return "rotate180";
		
		case COL2IM:
			return "col2im";
		
		case POOLING_PRE_RESHAPE:
			return "pooling_pre_reshape";
			
		case MAX_POOLING:
			return "maxpooling";
			
		case POOLING_POST_RESHAPE:
			return "pooling_post_reshape";
			
		case POOLING_BACKWARD_RESHAPE:
			return "pooling_backward_reshape";
			
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
				
		}
	}
	
	//CP instructions
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
	public String getInstructions(String input, String stride1, String stride2, String padding1, String padding2, 
			String input_shape1, String input_shape2, String input_shape3, String input_shape4,
			String filter_shape1, String filter_shape2, String filter_shape3, String filter_shape4,
			String output) throws LopsException {
		//only used for im2col and col2im
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		
		//rows, cols, byrow
		String[] inputX = new String[]{stride1, stride2, padding1, padding2, 
			 input_shape1, input_shape2, input_shape3, input_shape4,
			 filter_shape1, filter_shape2, filter_shape3, filter_shape4};
		for( int i=1; i<=(inputX.length); i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( ltmp.prepScalarInputOperand(getExecType()));
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	public String getInstructions(String input, String dout, String stride1, String stride2, String padding1, String padding2, 
			String input_shape1, String input_shape2, String input_shape3, String input_shape4,
			String filter_shape1, String filter_shape2, String filter_shape3, String filter_shape4,
			String output) throws LopsException {
		//only used for im2col and col2im
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		
		String[] inputX = new String[]{dout, stride1, stride2, padding1, padding2, 
			 input_shape1, input_shape2, input_shape3, input_shape4,
			 filter_shape1, filter_shape2, filter_shape3, filter_shape4};
		for( int i=1; i<=(inputX.length); i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( ltmp.prepScalarInputOperand(getExecType()));
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	public static ConvolutionTransform constructConvolutionTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.ConvolutionTransform ) {
				return (ConvolutionTransform)lop;
			}
		}
		ConvolutionTransform retVal = new ConvolutionTransform(input1, op, dt, vt);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal;
	}

	public static ConvolutionTransform constructConvolutionTransformLop(Lop input1, OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		
		for (Lop lop  : input1.getOutputs()) {
			if ( lop.type == Lop.Type.ConvolutionTransform ) {
				return (ConvolutionTransform)lop;
			}
		}
		ConvolutionTransform retVal = new  ConvolutionTransform(input1, op, dt, vt, et);
		retVal.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndColumn());
		return retVal; 
	}

 
}
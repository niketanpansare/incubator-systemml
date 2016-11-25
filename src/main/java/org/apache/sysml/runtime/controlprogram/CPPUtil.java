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

package org.apache.sysml.runtime.controlprogram;

import java.util.ArrayList;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.IntObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixNative;

import java.util.HashMap;

/**
 * 
 * 1. Compile the jar: 
 * mvn package
 * 
 * 2. Compile headers (only for development):
 * javah -classpath SystemML.jar org.apache.sysml.runtime.controlprogram.CPPUtil
 * 
 * 3. 
 */
public class CPPUtil {
	private ExecutionContext _ec;
	private HashMap<String, Integer> varToIDMap = new HashMap<String, Integer>();
	private int _globalVarCounter = 0;
	public static String [] SUPPORTED_OPCODES = { "ba+*" }; 
	public static enum SUPPORTED_BLOCKS { GENERIC, FOR }; 
	
	public CPPUtil(ExecutionContext ec) {
		this._ec = ec;
	}
	
	static {
		if(LibMatrixNative.loadingThread != null) {
			try {
				LibMatrixNative.loadingThread.join();
			} catch (InterruptedException e) {}
		}
	}
	
	private native void execute(int [] encodedBlock, int lenEncodedBlock, int numVarID);
	public static native void matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen);
	
	// ----------------------------------------------------
	// TODO:
	public double[] getDenseBlock(int matrixID) {
		return null;
	}
	public int getNumRows(int matrixID) {
		return -1;
	}
	public int getNumCols(int matrixID) {
		return -1;
	}
	// ----------------------------------------------------
	
	public boolean execute(ProgramBlock blk) throws DMLRuntimeException {
		int [] arr = encode(blk);
		if(arr == null) {
			return false;
		}
		else {
			execute(arr, arr.length, varToIDMap.size());
			return true;
		}
	}
	
	private int [] encode(ProgramBlock blk) throws DMLRuntimeException {
		ArrayList<Integer> encodedList = encode_pgm_blk(blk);
		if(encodedList != null)
			return ArrayUtils.toPrimitive(encodedList.toArray(new Integer[encodedList.size()]));
		else
			return null;
	}
	
	private static Integer getInstructionID(String instOpCode) {
		for(int i = 0; i < SUPPORTED_OPCODES.length; i++) {
			if(SUPPORTED_OPCODES[i].equals(instOpCode)) return i;
		}
		return null;
	}
	
	private Integer getVariableID(String variableName) {
		if(varToIDMap.containsKey(variableName)) {
			return varToIDMap.get(variableName);
		}
		else {
			Integer id = new Integer(_globalVarCounter);
			varToIDMap.put(variableName, id);
			_globalVarCounter++;
			return id;
		}
	}
	
	private ArrayList<Integer> encode_pgm_blk(ProgramBlock blk) throws DMLRuntimeException {
		if(!DMLScript.ENABLE_NATIVE_LOOP) return null;
		
		if(blk instanceof ParForProgramBlock) {
			return null;
		}
		else if(blk instanceof ForProgramBlock) {
			ArrayList<ProgramBlock> childBlocks = ((ForProgramBlock) blk).getChildBlocks();
			ArrayList<Integer> ret = new ArrayList<Integer>();
			
			String iterVarName = ((ForProgramBlock) blk).getIterablePredicateVars()[0];

			// evaluate from, to, incr only once (assumption: known at for entry)
			IntObject from = ((ForProgramBlock) blk).executePredicateInstructions( 1, ((ForProgramBlock) blk).getFromInstructions(), _ec );
			IntObject to   = ((ForProgramBlock) blk).executePredicateInstructions( 2, ((ForProgramBlock) blk).getToInstructions(), _ec );
			ArrayList<Instruction> incrementInstructions = ((ForProgramBlock) blk).getIncrementInstructions();
			IntObject incr = (incrementInstructions == null || incrementInstructions.isEmpty()) && ((ForProgramBlock) blk).getIterablePredicateVars()[3]==null ? 
					new IntObject((from.getLongValue()<=to.getLongValue()) ? 1 : -1) :
						((ForProgramBlock) blk).executePredicateInstructions( 3, incrementInstructions, _ec );
			
			ret.add(SUPPORTED_BLOCKS.FOR.ordinal());
			ret.add(getVariableID(iterVarName));		
			ret.add((int)from.getLongValue());
			ret.add((int)to.getLongValue());
			ret.add((int)incr.getLongValue());
			ret.add((int)childBlocks.size());
			for(ProgramBlock childBlock : childBlocks) {
				ArrayList<Integer> encodedChildValues = encode_pgm_blk(childBlock);
				if(encodedChildValues == null) {
					return null;
				}
				else {
					ret.addAll(encodedChildValues);
				}
			}
			return ret;
		}
		else if(blk instanceof ExternalFunctionProgramBlock) {
			return null;
		}
		else if(blk instanceof FunctionProgramBlock) {
			return null;
		}
		else if(blk instanceof IfProgramBlock) {
			return null;
		}
		else if(blk instanceof WhileProgramBlock) {
			return null;
		}
		else {
			// Generic Program Block
			ArrayList<Integer> ret = new ArrayList<Integer>();
			ret.add(SUPPORTED_BLOCKS.GENERIC.ordinal());
			ret.add(blk.getInstructions().size());
			for(Instruction inst : blk.getInstructions()) {
				Integer id = getInstructionID(inst.getExtendedOpcode());
				if(id == null) {
					return null;
				}
				else {
					// Encode instruction
					ret.add(id);
					ArrayList<Integer> encodedParams = inst.getEncodedCPPParameters();
					if(encodedParams != null) {
						ret.addAll(encodedParams); 
					}
				}
			}
			return ret;
		}
	}
}

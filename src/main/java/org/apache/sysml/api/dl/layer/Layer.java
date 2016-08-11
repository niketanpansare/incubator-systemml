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
package org.apache.sysml.api.dl.layer;

import java.util.ArrayList;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.Barista.DMLPhase;
import org.apache.sysml.api.dl.utils.DLUtils;
import org.apache.sysml.api.dl.utils.MathUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public abstract class Layer {
	private static int id = 1;
	public LayerParameter param;
	
	public String gradientPrefix = "grad_";
	public String updatePrefix = "upd_";
	
	// Used by next layer for forward propogation
//	public String outputVar; // to top layer
//	public String getBottomLayerOutputVar() {
//		return bottom.get(0).outputVar;
//	}
	public String deltaVar; // from top layer
	
	public String weightVar;
	public String biasVar;
	
	public int layerID;
	public ArrayList<String> output_shape = new ArrayList<String>();
	
	private ArrayList<String> getOneElemAL(String val) {
		ArrayList<String> ret = new ArrayList<String>();
		ret.add(val);
		return ret;
	}
	
	public Layer(LayerParameter param, String outputVar1Prefix, String outputVar2Prefix) throws DMLRuntimeException {
		this.param = param;
		this.layerID = (Layer.id++);
		assertTop(2);
		
		if(Barista.blobToDMLVariableMapping.containsKey(param.getTop(0)))
			throw new DMLRuntimeException("Error while generating DML code. Already generated output variable for " + param.getTop(0));
				
		if(Barista.blobToDMLVariableMapping.containsKey(param.getTop(1)))
			throw new DMLRuntimeException("Error while generating DML code. Already generated output variable for " + param.getTop(1));
		
		if(this instanceof DataLayer) {
			if(Barista.currentPhase == DMLPhase.TRAIN)
				Barista.blobToDMLVariableMapping.put(param.getTop(0), getOneElemAL(outputVar1Prefix + Barista.trainingVarsuffix));
			else if(Barista.currentPhase == DMLPhase.TEST)
				Barista.blobToDMLVariableMapping.put(param.getTop(0), getOneElemAL(outputVar1Prefix + Barista.testVarsuffix));
			else
				throw new DMLRuntimeException("Should not create layer in validation phase");
		}
		else
			Barista.blobToDMLVariableMapping.put(param.getTop(0), getOneElemAL(outputVar1Prefix + layerID));
		
		if(this instanceof DataLayer) {
			if(Barista.currentPhase == DMLPhase.TRAIN)
				Barista.blobToDMLVariableMapping.put(param.getTop(1), getOneElemAL(outputVar2Prefix + Barista.trainingVarsuffix));
			else if(Barista.currentPhase == DMLPhase.TEST)
				Barista.blobToDMLVariableMapping.put(param.getTop(1), getOneElemAL(outputVar2Prefix + Barista.testVarsuffix));
			else
				throw new DMLRuntimeException("Should not create layer in validation phase");
		}
		else
			Barista.blobToDMLVariableMapping.put(param.getTop(1), getOneElemAL(outputVar2Prefix + layerID));
		
		this.deltaVar = "delta_" + layerID;
	}
	
	public Layer(LayerParameter param, String outputVarPrefix) throws DMLRuntimeException {
		this.param = param;
		this.layerID = (Layer.id++);
		assertTop(1);
		if(Barista.blobToDMLVariableMapping.containsKey(param.getTop(0)))
			throw new DMLRuntimeException("Error while generating DML code. Already generated output variable for " + param.getTop(0));
		
		if(this instanceof DataLayer) {
			throw new DMLRuntimeException("Expected two top layers for DataLayer");
		}
		else
			Barista.blobToDMLVariableMapping.put(param.getTop(0), getOneElemAL(outputVarPrefix + layerID));
		
		this.deltaVar = "delta_" + layerID;
	}
	protected void assertTop(int expectedNumTop) throws DMLRuntimeException {
		if(param.getTopCount() != expectedNumTop) {
			throw new DMLRuntimeException("Expected exactly " + expectedNumTop + " top for " + getPrettyLayerName() + " but found " + param.getTopCount());
		}
	}
	protected void assertBottom(int expectedNumBottom) throws DMLRuntimeException {
		if(param.getBottomCount() != expectedNumBottom) {
			throw new DMLRuntimeException("Expected exactly " + expectedNumBottom + " bottom for " + getPrettyLayerName() + " but found " + param.getBottomCount());
		}
	}
	// Throws exception if used for multi-top layers
	public String getTopDMLVar()  throws DMLRuntimeException {
		assertTop(1);
		ArrayList<String> ret = Barista.blobToDMLVariableMapping.get(param.getTop(0));
		if(ret == null) {
			throw new DMLRuntimeException("No output variable associated with the top variable.");
		}
		if(ret.size() != 1) {
			throw new DMLRuntimeException("More than 1 output variable associated with the top variable.");
		}
		return ret.get(0);
	}
	// Potentially unsafe method ... only use for DataLayer
	public String getTopDMLVar(int i)  throws DMLRuntimeException {
		ArrayList<String> ret = Barista.blobToDMLVariableMapping.get(param.getTop(i));
		if(ret == null) {
			throw new DMLRuntimeException("No output variable associated with the top variable:" + param.getTop(i));
		}
		if(ret.size() != 1) {
			throw new DMLRuntimeException("More than 1 output variable associated with the top variable.");
		}
		return ret.get(0);
	}
	// Throws exception if used for multi-bottom layers
	public String getBottomDMLVar()  throws DMLRuntimeException {
		assertBottom(1);
		ArrayList<String> ret = Barista.blobToDMLVariableMapping.get(param.getBottom(0));
		if(ret == null) {
			throw new DMLRuntimeException("No variable associated with the bottom variable for " + getPrettyLayerName());
		}
		if(ret.size() != 1) {
			throw new DMLRuntimeException("More than 1 variable associated with the bottom variable for " + getPrettyLayerName());
		}
		return ret.get(0);
	}
	// Potentially unsafe method ... only use for SoftmaxLossLayer
	public String getBottomDMLVar(int i)  throws DMLRuntimeException {
		ArrayList<String> ret = Barista.blobToDMLVariableMapping.get(param.getBottom(i));
		if(ret == null) {
			throw new DMLRuntimeException("No variable associated with the bottom variable:" + param.getBottom(i));
		}
		if(ret.size() != 1) {
			throw new DMLRuntimeException("More than 1 output variable associated with the bottom variable for " + getPrettyLayerName());
		}
		return ret.get(0);
	}
	
	public abstract void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException;
	public abstract void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException;
	public abstract void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException;
	public abstract String generateFinalizeDML() throws DMLRuntimeException;
	public abstract void updateOutputShape()  throws DMLRuntimeException;
	
	protected String getInputShape() throws DMLRuntimeException {
		return "input_shape=[" + getBottomLayerOutputShape(0) + "," + getBottomLayerOutputShape(1) + "," + getBottomLayerOutputShape(2) + "," + getBottomLayerOutputShape(3) + "]";
	}
	
	public String toString() {
		String ret = param.getName() + " <- [";
		for(int i = 0; i < param.getBottomCount(); i++) {
			 ret += " " + param.getBottom(i);
		}
		ret += "]";
		return ret;
	}
	
	public String getOutputShape(int index) throws DMLRuntimeException {
		String ret = "??";
		try {
			ret = output_shape.get(index);
		} catch(Exception e) {
			throw new DMLRuntimeException("Output shape is not available for the " + param.getType() + " layer " + param.getName());
		}
		return ret;
	}
	
	public String getBottomLayerDeltaVar() throws DMLRuntimeException {
		ArrayList<Layer> layers = DLUtils.findLayerWithTop(param.getBottom(0));
		if(layers.size() != 1) {
			throw new DMLRuntimeException("Expected only 1 bottom layer"); 
		}
		return layers.get(0).deltaVar;
	}
	
	public String getBottomLayerOutputShape(int index) throws DMLRuntimeException {
		String ret = "??";
		try {
			ArrayList<Layer> layers = DLUtils.findLayerWithTop(param.getBottom(0));
			if(layers.size() != 1) {
				throw new DMLRuntimeException("Expected only 1 bottom layer"); 
			}
			ret = layers.get(0).getOutputShape(index);
		} catch(Exception e) {
			throw new DMLRuntimeException("Output shape is not available for the bottom layer of " + param.getType() + " layer " + param.getName());
		}
		return ret;
	}
	
	public String getPrettyLayerName() {
		return "\"" + param.getType() + " layer:" + param.getName() + "\"";
	}
	
	protected void printSetupHeader(StringBuilder dmlScript) {
		dmlScript.append("\n# Setup " + getPrettyLayerName() + " \n");
	}
	protected void printForwardHeader(TabbedStringBuilder dmlScript) {
		dmlScript.append("\n");
		dmlScript.append("# Perform forward pass on " + getPrettyLayerName() + " \n");
	}
	protected void printBackwardHeader(TabbedStringBuilder dmlScript) {
		dmlScript.append("\n");
		dmlScript.append("# Perform backward pass on " + getPrettyLayerName() + " \n");
	}
	
	// Utility functions
	protected String matmult(String left, String right) {
		return left + " %*% " + right;
	}
	protected String mult(String left, String right) {
		return left + " * " + right;
	}
	protected String divide(String left, String right) {
		return left + " / " + right;
	}
	// Scalar multiply
	protected String smult(String left, String right) {
		return MathUtils.scalarMultiply(left, right);
	}
	protected String toInt(String val) {
		return MathUtils.toInt(val);
	}
	protected String add(String left, String right) {
		return left + " + " + right;
	}
	protected String ssubtract(String left, String right) {
		return MathUtils.scalarSubtraction(left, right);
	}
	protected String subtract(String left, String right) {
		return left + " - " + right;
	}
	protected String subtract(String left, String right, boolean inBrackets) {
		if(inBrackets)
			return "(" + subtract(left, right) + ")";
		else
			return subtract(left, right);
	}
	protected String getShape(String shape1, String shape2, String shape3, String shape4) {
		return "[" + shape1 + ", " + shape2 + ", " + shape3 + ", " + shape4 + "]";
	}
	protected String getShape(int shape1, String shape2, int shape3, int shape4) {
		return getShape("" + shape1, shape2, "" + shape3, "" + shape4);
	}
	protected String matrix(String init, String rows, String cols) {
		return "matrix(" + init + ", rows=" + rows + ", cols=" + cols + ")";
	}
	protected String matrix(double init, String rows, String cols) {
		return "matrix(" + init + ", rows=" + rows + ", cols=" + cols + ")";
	}
	protected String matrix(double init, int rows, String cols) {
		return "matrix(" + init + ", rows=" + rows + ", cols=" + cols + ")";
	}
	protected String matrix(double init, String rows, int cols) {
		return "matrix(" + init + ", rows=" + rows + ", cols=" + cols + ")";
	}
	protected String assign(String lhs, String rhs, String comment) {
		return lhs + " = " + rhs + "; #" + comment + "\n";
	}
	protected String assign(String lhs, String rhs) {
		return lhs + " = " + rhs + ";\n";
	}
	protected String layerVar(String prefix) {
		return prefix + "_" + layerID;
	}
	protected String nrow(String var) {
		return "nrow(" + var + ")";
	}
	protected String ncol(String var) {
		return "ncol(" + var + ")";
	}
	protected String read(String fileName) {
		return "read(" + fileName + ")";
	}
	protected String inQuotes(String val) {
		return "\"" + val + "\"";
	}
	protected String ifdef(String commandlineArg, String defaultVal) {
		return "ifdef($" + commandlineArg + "," + defaultVal + ")";
	}
	protected String ifLoop(String pred) {
		return "if(" + pred + ") {\n";
	}
	protected String seq(int start, String end, int incr) {
		return "seq(" + start + ", " + end + ", " + incr + ")";
	}
	protected String table(String v1, String v2, String v3, String v4) {
		return "table(" +  v1 + ", " + v2 + ", " + v3 + ", " + v4 + ")";
	}
	protected String t(String val) {
		return "t(" + val + ")";
	}
	protected String rand(String rows, String cols, String min, String max, String pdf, String sparsity) {
		return "rand(rows=" + rows + ", cols=" + cols + ", min=" + min + ", max=" + max + ", pdf=" + inQuotes(pdf)
				+ ", sparsity=" + sparsity + ")"; 
	}
	protected String gt(String lhs, String rhs, boolean inBrackets) {
		if(inBrackets)
			return "(" + lhs + " > " + rhs + ")";
		else
			return lhs + " > " + rhs;
	}
	protected String max(String v1, String v2) {
		return "max(" + v1 + ", " + v2 +")";
	}
	protected String exp(String v1) {
		return "exp(" + v1 + ")";
	}
	protected String rowSums(String v1) {
		return "rowSums(" + v1 + ")";
	}
	protected String colSums(String v1) {
		return "colSums(" + v1 + ")";
	}
	protected String print(String v1) {
		return "print(" + v1 + ");\n";
	}
	protected String sum(String v1) {
		return "sum(" + v1 + ")";
	}
	protected String log(String v1) {
		return "log(" + v1 + ")";
	}
}
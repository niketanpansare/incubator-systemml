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
	public String outputVar; // to top layer
	public String getBottomLayerOutputVar() {
		return bottom.get(0).outputVar;
	}
	public String deltaVar; // from top layer
	public String weightVar;
	public String biasVar;
	
	public int layerID;
	public ArrayList<String> output_shape = new ArrayList<String>();
	public ArrayList<Layer> bottom = new ArrayList<Layer>();
	
	public Layer(LayerParameter param, String outputVarPrefix) {
		this.param = param;
		this.layerID = (Layer.id++);
		if(this instanceof DataLayer)
			this.outputVar = outputVarPrefix;
		else
			this.outputVar = outputVarPrefix + layerID;
		this.deltaVar = "delta_" + layerID;
	}
	
	public abstract void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException;
	public abstract void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException;
	public abstract void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException;
	public abstract String generateFinalizeDML() throws DMLRuntimeException;
	public abstract void updateOutputShape()  throws DMLRuntimeException;
	
	protected String getInputShape() {
		return "input_shape=[" + getBottomLayerOutputShape(0) + "," + getBottomLayerOutputShape(1) + "," + getBottomLayerOutputShape(2) + "," + getBottomLayerOutputShape(3) + "]";
	}
	
	protected void checkInput() throws DMLRuntimeException {
		if(bottom.size() == 0) {
			throw new DMLRuntimeException("The layer " + param.getName() + " cannot be the bottom-most layer");
		}
		else if(bottom.size() > 1) {
			throw new DMLRuntimeException("Multi-input " + param.getName() + " is not implemented");
		}
	}
	
	public String toString() {
		String ret = param.getName() + " <- [";
		for(int i = 0; i < param.getBottomCount(); i++) {
			 ret += " " + param.getBottom(i);
		}
		ret += "]";
		return ret;
	}
	
	public String getOutputShape(int index) {
		String ret = "??";
		try {
			ret = output_shape.get(index);
		} catch(Exception e) {}
		return ret;
	}
	
	public String getBottomLayerOutputShape(int index) {
		String ret = "??";
		try {
			ret = bottom.get(0).getOutputShape(index);
		} catch(Exception e) {}
		return ret;
	}
	
	protected void printSetupHeader(StringBuilder dmlScript) {
		dmlScript.append("\n# Setup the " + param.getType() + " layer " + param.getName() + " \n");
	}
	protected void printForwardHeader(TabbedStringBuilder dmlScript) {
		dmlScript.append("\n");
		dmlScript.append("# Perform forward pass on the " + param.getType() + " layer " + param.getName() + " \n");
	}
	protected void printBackwardHeader(TabbedStringBuilder dmlScript) {
		dmlScript.append("\n");
		dmlScript.append("# Perform backward pass on the " + param.getType() + " layer " + param.getName() + " \n");
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
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
package org.apache.sysml.api.dl.utils;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.FillerParameter;
import caffe.Caffe.FillerParameter.VarianceNorm;

//For more details see https://github.com/BVLC/caffe/blob/master/include/caffe/filler.hpp
public class FillerUtils {
	
	public static String getFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		if(weightFiller.getType().equalsIgnoreCase("constant")) {
			return getConstantFiller(lhsVar, shape, weightFiller, numTabs);
		}
		else if(weightFiller.getType().equalsIgnoreCase("uniform")) {
			return getUniformFiller(lhsVar, shape, weightFiller, numTabs);
		}
		else if(weightFiller.getType().equalsIgnoreCase("gaussian")) {
			return getGaussianFiller(lhsVar, shape, weightFiller, numTabs);
		}
		else if(weightFiller.getType().equalsIgnoreCase("xavier")) {
			return getXavierFiller(lhsVar, shape, weightFiller, numTabs);
		}
		else if(weightFiller.getType().equalsIgnoreCase("msra")) {
			return getMSRAFiller(lhsVar, shape, weightFiller, numTabs);
		}
		
		throw new DMLRuntimeException("Unsupported type of weight filler: " + weightFiller.getType() 
				+ ". The available options are { constant, uniform, gaussian, xavier, msra }");
	}
	
	private static String getConstantFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		if(shape.size() == 4)
			return tabs + lhsVar + " = matrix(" + weightFiller.getValue() + ", rows=" + MathUtils.toInt(shape.get(0)) 
				+ ", cols=" + MathUtils.toInt(MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3)))
				+ ");\n";
		else if(shape.size() == 2)
			return tabs + lhsVar + " = matrix(" + weightFiller.getValue() + ", rows=" + MathUtils.toInt(shape.get(0)) 
					+ ", cols=" + MathUtils.toInt(shape.get(1))
					+ ");\n";
		throw new DMLRuntimeException("Expected the shape to be of size 2 or 4");
	}
	
	private static String getUniformFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		if(shape.size() != 4 ) {
			throw new DMLRuntimeException("Expected the shape to be of size 4");
		}
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		return tabs + lhsVar + " = rand(rows=" + MathUtils.toInt(shape.get(0)) 
				+ ", cols=" + MathUtils.toInt(MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3)))
				+ ", min=" + weightFiller.getMin() + ", max=" + weightFiller.getMax() + ", pdf=\"uniform\");\n";
	}
	
	private static String getGaussianFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		if(shape.size() != 4 ) {
			throw new DMLRuntimeException("Expected the shape to be of size 4");
		}
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		
		// scale-location transformation: 
		// if x ~ Normal(0, var=1), then s*x + m ~ N(m, var=s^2)
		return tabs + lhsVar + " = "  + weightFiller.getMean() + " + " + weightFiller.getStd() 
				+ " * rand(rows=" + MathUtils.toInt(shape.get(0)) 
				+ ", cols=" + MathUtils.toInt(MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3)))
				+ ", pdf=\"normal\");\n";
	}
	
	private static String getXavierFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		if(shape.size() != 4 ) {
			throw new DMLRuntimeException("Expected the shape to be of size 4");
		}
		String fan_in = MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3));
		String fan_out = MathUtils.scalarMultiply(shape.get(0), shape.get(2), shape.get(3));
		String n = fan_in;
		if(weightFiller.getVarianceNorm() == VarianceNorm.AVERAGE) {
			n = MathUtils.scalarDivision(MathUtils.scalarAddition(fan_in, fan_out), "2.0");
		}
		else if(weightFiller.getVarianceNorm() == VarianceNorm.FAN_OUT) {
			n = fan_out;
		}
		String scale = MathUtils.sqrt(MathUtils.scalarDivision("3.0", n));
		
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		
		return tabs + "# Using Xavier filler. See [Bengio and Glorot 2010]: Understanding the difficulty of training deep feedforward neuralnetworks.\n"
				+ tabs + lhsVar + " = rand(rows=" + MathUtils.toInt(shape.get(0)) + ", cols=" + MathUtils.toInt(fan_in)  
				+ ", min=-" + scale + ", max=" + scale + ", pdf=\"uniform\");\n"; 
	}
	
	
	private static String getMSRAFiller(String lhsVar, ArrayList<String> shape, FillerParameter weightFiller, int numTabs) throws DMLRuntimeException {
		if(shape.size() != 4 ) {
			throw new DMLRuntimeException("Expected the shape to be of size 4");
		}
		String fan_in = MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3));
		String fan_out = MathUtils.scalarMultiply(shape.get(0), shape.get(2), shape.get(3));
		String n = fan_in;
		if(weightFiller.getVarianceNorm() == VarianceNorm.AVERAGE) {
			n = MathUtils.scalarDivision(MathUtils.scalarAddition(fan_in, fan_out), "2.0");
		}
		else if(weightFiller.getVarianceNorm() == VarianceNorm.FAN_OUT) {
			n = fan_out;
		}
		String std = MathUtils.sqrt(MathUtils.scalarDivision("2.0", n));
		
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		
		// scale-location transformation: 
		// if x ~ Normal(0, var=1), then s*x + m ~ N(m, var=s^2)
		return tabs + "# Using MSRA filler. See [Saxe, McClelland, and Ganguli 2013 (v3)].\n"
				+ tabs + lhsVar + " = " + std 
				+ " * rand(rows=" + MathUtils.toInt(shape.get(0)) 
				+ ", cols=" + MathUtils.toInt(MathUtils.scalarMultiply(shape.get(1), shape.get(2), shape.get(3)))
				+ ", pdf=\"normal\");\n";
	}
	
	
	public static String getSimpleMSRAFiller(String lhsVar, ArrayList<String> shape, int numTabs) throws DMLRuntimeException {
		if(shape.size() != 2 ) {
			throw new DMLRuntimeException("Expected the shape to be of size 2");
		}
		String n = shape.get(0);
		
		String std = MathUtils.sqrt(MathUtils.scalarDivision("2.0", n));
		
		String tabs = "";
		for(int i = 0; i < numTabs; i++) {
			tabs += "\t";
		}
		
		// scale-location transformation: 
		// if x ~ Normal(0, var=1), then s*x + m ~ N(m, var=s^2)
		return tabs + "# Using MSRA filler. See [Saxe, McClelland, and Ganguli 2013 (v3)].\n"
				+ tabs + lhsVar + " = " + std 
				+ " * rand(rows=" + MathUtils.toInt(shape.get(0)) 
				+ ", cols=" + shape.get(1)
				+ ", pdf=\"normal\");\n";
	}
}

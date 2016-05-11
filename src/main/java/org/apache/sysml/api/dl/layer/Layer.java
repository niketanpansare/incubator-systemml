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

import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public abstract class Layer {
	private static int id = 1;
	public LayerParameter param;
	
	public String gradientPrefix = "grad_";
	public String updatePrefix = "upd_";
	
	// Used by next layer for forward propogation
	public String outputVar; // to top layer
	public String deltaVar; // from top layer
	public String weightVar;
	public String biasVar;
	
	public int layerID;
	public ArrayList<String> output_shape = new ArrayList<String>();
	public ArrayList<Layer> bottom = new ArrayList<Layer>();
	
	// TODO:
	// Used by previous layer for backward propogation
	
	public Layer(LayerParameter param, String outputVarPrefix) {
		this.param = param;
		this.layerID = (Layer.id++);
		this.outputVar = outputVarPrefix + layerID;
		this.deltaVar = "delta_" + layerID;
	}
	
	public abstract String getSetupDML() throws DMLRuntimeException;
	public abstract String getForwardDML() throws DMLRuntimeException;
	public abstract String getBackwardDML() throws DMLRuntimeException;
	public abstract String getFinalizeDML() throws DMLRuntimeException;
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
	
}
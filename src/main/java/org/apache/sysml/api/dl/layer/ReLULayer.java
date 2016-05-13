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

import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class ReLULayer extends Layer {

	public ReLULayer(LayerParameter param) {
		super(param, "H_");
		// TODO Auto-generated constructor stub
	}

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		checkInput();
		printSetupHeader(dmlScript);	
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printForwardHeader(dmlScript);
		dmlScript.append(assign(outputVar, max("0", getBottomLayerOutputVar())));
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printBackwardHeader(dmlScript);
		if(bottom.size() != 1) {
			throw new DMLRuntimeException("Multiple bottom layers not implemented");
		}
		dmlScript.append(assign(bottom.get(0).deltaVar, mult(gt(bottom.get(0).outputVar, "0", true), deltaVar)));
	}

	@Override
	public String generateFinalizeDML() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		output_shape.clear();
		output_shape.add(getBottomLayerOutputShape(0));
		output_shape.add(getBottomLayerOutputShape(1));
		output_shape.add(getBottomLayerOutputShape(2));
		output_shape.add(getBottomLayerOutputShape(3));
	}

}
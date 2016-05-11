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

import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.DropoutParameter;
import caffe.Caffe.LayerParameter;

public class DropoutLayer extends Layer {

	DropoutParameter dropParam;
	String maskVar;
	public DropoutLayer(LayerParameter param) {
		super(param, "H_");
		maskVar = "mask_" + layerID;
		dropParam = param.getDropoutParam();
	}

	@Override
	public String getSetupDML() throws DMLRuntimeException {
		return null;
	}

	@Override
	public String getForwardDML() throws DMLRuntimeException {
		return maskVar + " = (rand(rows=1, cols=ncol(" + bottom.get(0).outputVar + "), min=0, max=1, pdf=\"uniform\", "
				+ "sparsity=" + dropParam.getDropoutRatio() + ") > 0) / 0.5;\n"
				+ "\t" + outputVar + " = " + bottom.get(0).outputVar + " * " + maskVar + ";";
	}

	@Override
	public String getBackwardDML() throws DMLRuntimeException {
		if(bottom.size() != 1) {
			throw new DMLRuntimeException("Multiple bottom layers not implemented");
		}
		return bottom.get(0).deltaVar + " = (" + bottom.get(0).outputVar + " > 0) * " + maskVar  + " * " + deltaVar + ";";
	}

	@Override
	public String getFinalizeDML() throws DMLRuntimeException {
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

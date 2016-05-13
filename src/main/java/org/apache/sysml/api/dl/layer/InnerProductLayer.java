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
import org.apache.sysml.api.dl.utils.FillerUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.InnerProductParameter;
import caffe.Caffe.LayerParameter;

public class InnerProductLayer extends Layer {

	InnerProductParameter innerParam;
	
	public InnerProductLayer(LayerParameter param) {
		super(param, "H_");
		weightVar = "w_" + layerID;
		innerParam = param.getInnerProductParam();
		if(innerParam.hasBiasFiller()) {
			biasVar = "bias_" + layerID;
		}
	}

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		checkInput();
		
		printSetupHeader(dmlScript);
		
		ArrayList<String> shape = new ArrayList<String>();
		shape.add("" + innerParam.getNumOutput());
		shape.add(getBottomLayerOutputShape(1));
		shape.add(getBottomLayerOutputShape(2));
		shape.add(getBottomLayerOutputShape(3));
		
		dmlScript.append(FillerUtils.getFiller(weightVar, shape, innerParam.getWeightFiller(), 0));
		
		if(innerParam.hasBiasFiller()) {
			ArrayList<String> bias_shape = new ArrayList<String>();
			bias_shape.add("" + innerParam.getNumOutput());
			bias_shape.add("1");
			// Transposing to produce Prithvi's script
			dmlScript.append(FillerUtils.getFiller(biasVar, bias_shape, innerParam.getBiasFiller(), 0));
			dmlScript.append(assign(weightVar, t(weightVar)));
			dmlScript.append(assign(biasVar, t(biasVar)));
		}
		else {
			dmlScript.append(assign(weightVar, t(weightVar)));
		}
		
		if(Barista.USE_MOMENTUM) {
			dmlScript.append(assign(updatePrefix + weightVar, matrix(0, nrow(weightVar), ncol(weightVar))));
			dmlScript.append(assign(updatePrefix + biasVar, matrix(0, nrow(biasVar), ncol(biasVar))));
		}
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) {
		printForwardHeader(dmlScript);
		dmlScript.append(assign(outputVar, add(matmult(getBottomLayerOutputVar(), weightVar), biasVar)));
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printBackwardHeader(dmlScript);
		dmlScript.append(assign(gradientPrefix + weightVar, matmult(t(getBottomLayerOutputVar()), deltaVar)));
		if(innerParam.hasBiasFiller()) {
			dmlScript.append(assign(gradientPrefix + biasVar, colSums(deltaVar)));
		}
		dmlScript.append(assign(bottom.get(0).deltaVar, matmult(deltaVar, t(weightVar))));
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
		output_shape.add("" + innerParam.getNumOutput());
		output_shape.add("1");
		output_shape.add("1");
	}

}
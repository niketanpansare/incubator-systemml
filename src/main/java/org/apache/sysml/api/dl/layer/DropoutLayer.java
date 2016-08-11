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

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.Barista.DMLPhase;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.DropoutParameter;
import caffe.Caffe.LayerParameter;

public class DropoutLayer extends Layer {

	DropoutParameter dropParam;
	String maskVar;
	
	public DropoutLayer(LayerParameter param) throws DMLRuntimeException {
		super(param, "H_");
		maskVar = "mask_" + layerID;
		dropParam = param.getDropoutParam();
	}

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printForwardHeader(dmlScript);
		if(Barista.currentPhase == DMLPhase.TRAIN) {
			String ratio = "" + dropParam.getDropoutRatio();
			String randCall = rand("1", ncol(getBottomDMLVar()), "0", "1", "uniform", ratio);
			dmlScript.append(assign(maskVar, divide(gt(randCall, "0", true), ssubtract("1", ratio))));
			dmlScript.append(assign(getTopDMLVar(), mult(getBottomDMLVar(), maskVar)));
		}
		else {
			dmlScript.append(assign(getTopDMLVar(), getBottomDMLVar()));
		}
		
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printBackwardHeader(dmlScript);
		assertBottom(1);
		
		// (delta > 0) * mask 
		dmlScript.append(assign(getBottomLayerDeltaVar(), mult(mult(gt(getBottomDMLVar(), "0", true), maskVar), deltaVar)));
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

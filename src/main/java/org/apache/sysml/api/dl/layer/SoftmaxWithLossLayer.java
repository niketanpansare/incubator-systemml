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
import org.apache.sysml.api.dl.utils.DLUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class SoftmaxWithLossLayer extends Layer {
	String bottomActivationVar;
	String labelVar;
	
	public SoftmaxWithLossLayer(LayerParameter param) throws DMLRuntimeException {
		super(param, "H_");
		if(param.getBottomCount() != 2) 
			throw new DMLRuntimeException("Expected exactly two bottom layers for " + getPrettyLayerName());
		
		// Assumption: First bottom layer is prediction and the second one is the label
		bottomActivationVar = getBottomDMLVar(0);
		labelVar = getBottomDMLVar(1);
	}

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		// checkInput(); -> Allow multiple inputs
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printForwardHeader(dmlScript);
		String tmp = "unNormalizedProbs_" + layerID;
		dmlScript.append(assign(tmp, exp(bottomActivationVar)));
		dmlScript.append(assign(getTopDMLVar(), divide(tmp, rowSums(tmp))));
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		dmlScript.append("#" + print(inQuotes("ITER=") + " + iter + " + inQuotes(" OBJ=") + " + " 
				+ sum(mult(labelVar, log(getTopDMLVar()))))); 
				
		printBackwardHeader(dmlScript);
		assertBottom(2);
		dmlScript.append(assign(getBottomLayerDeltaVar(), subtract(labelVar, getTopDMLVar())));
	}

	@Override
	public String generateFinalizeDML() throws DMLRuntimeException {
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		// TODO Auto-generated method stub

	}
	
}
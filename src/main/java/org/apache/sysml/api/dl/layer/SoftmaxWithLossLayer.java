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
import org.apache.sysml.api.dl.utils.DLUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class SoftmaxWithLossLayer extends Layer {
	public SoftmaxWithLossLayer(LayerParameter param) {
		super(param, "H_");
	}

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		// checkInput(); -> Allow multiple inputs
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printForwardHeader(dmlScript);
		String tmp = "unNormalizedProbs_" + layerID;
		dmlScript.append(assign(tmp, exp(getBottomOutputVarForSoftMax())));
		dmlScript.append(assign(outputVar, divide(tmp, rowSums(tmp))));
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		dmlScript.append("#" + print(inQuotes("ITER=") + " + iter + " + inQuotes(" OBJ=") + " + " 
				+ sum(mult(getLabelVar(), log(outputVar))))); 
				
		printBackwardHeader(dmlScript);
		if(bottom.size() != 2) {
			throw new DMLRuntimeException("Expected 2 bottom layers for SoftmaxWithLoss layer");
		}
		dmlScript.append(assign(bottom.get(0).deltaVar, subtract(getLabelVar(), outputVar)));
	}

	@Override
	public String generateFinalizeDML() throws DMLRuntimeException {
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		// TODO Auto-generated method stub

	}
	
	private String getLabelVar() throws DMLRuntimeException {
		Layer labelLayer = getBottomLabelLayer();
		if(labelLayer instanceof DataLayer && ((DataLayer)labelLayer).labelVar != null) 
			return "oneHotEncoded_" + ((DataLayer)labelLayer).labelVar + Barista.trainingVarsuffix;
		else 
			throw new DMLRuntimeException("Expected labelVar to be set in DataLayer");
	}
	
	private String getBottomOutputVarForSoftMax() {
		if(DLUtils.topLayer.get("label") != bottom.get(0))
			return bottom.get(0).outputVar;
		else
			return bottom.get(1).outputVar;
	}
	
	public Layer getBottomLabelLayer() throws DMLRuntimeException {
		if(DLUtils.topLayer.get("label") == bottom.get(0)) {
			return bottom.get(0);
		}
		else if(DLUtils.topLayer.get("label") == bottom.get(1)) {
			return bottom.get(1);
		}
		else
			throw new DMLRuntimeException("Expected a layer with top: label");
	}

}
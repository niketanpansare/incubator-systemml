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
import org.apache.sysml.api.dl.utils.FillerUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.LayerParameter;


public class ConvolutionLayer extends Layer {

	int pad_h = 0; int pad_w = 0; int stride_h = 1; int stride_w = 1;
	int kernel_h = -1; int kernel_w = -1; int numFilter = -1;
	ConvolutionParameter convParam;
	String oneVar;
	String nColOneVar;
	
	boolean useBias = true;
	
	public ConvolutionLayer(LayerParameter param) throws DMLRuntimeException {
		super(param, "H_");
		weightVar = "w_" + layerID;
		convParam = param.getConvolutionParam();
		if(!convParam.hasBiasFiller()) {
			throw new DMLRuntimeException("bias filler required in the initial implementation");
		}
		if(useBias) {
			biasVar = "bias_" + layerID;
			oneVar = "ones_" + layerID;
		}
		if(convParam.hasGroup())
			System.out.println("WARNING: Ignoring group in convolution  in the initial implementation");
		
		numFilter = convParam.getNumOutput();
		
		pad_h = 0; pad_w = 0;
		if(convParam.hasPadH())
			pad_h = convParam.getPadH();
		else if(convParam.getPadCount() > 0)
			pad_h = convParam.getPad(0);
		if(convParam.hasPadW())
			pad_w = convParam.getPadW();
		else if(convParam.getPadCount() > 0)
			pad_w = convParam.getPad(0);
		
		if(convParam.hasStrideH())
			stride_h = convParam.getStrideH();
		else if(convParam.getStrideCount() > 0)
			stride_h = convParam.getStride(0);
		if(convParam.hasStrideW())
			stride_w = convParam.getStrideW();
		else if(convParam.getStrideCount() > 0)
			stride_w = convParam.getStride(0);
		
		if(convParam.hasKernelH())
			kernel_h = convParam.getKernelH();
		else if(convParam.getKernelSizeCount() > 0)
			kernel_h = convParam.getKernelSize(0);
		
		if(convParam.hasKernelW())
			kernel_w = convParam.getKernelW();
		else if(convParam.getKernelSizeCount() > 0)
			kernel_w = convParam.getKernelSize(0);
	}
	
	String filterShape;
	String convParamStr = "";

	@Override
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		printSetupHeader(dmlScript);
		dmlScript.append(assign(layerVar("P"), DLUtils.getP_DML(getBottomLayerOutputShape(2), pad_h, kernel_h, stride_h), "Output feature height"));
		dmlScript.append(assign(layerVar("Q"), DLUtils.getP_DML(getBottomLayerOutputShape(3), pad_w, kernel_w, stride_w), "Output feature width"));
		
		String numChannels = getBottomLayerOutputShape(1);
		ArrayList<String> shape = new ArrayList<String>();
		shape.add("" + numFilter);
		shape.add("" + numChannels);
		shape.add("" + kernel_h);
		shape.add("" + kernel_w);
		filterShape = "filter_shape=" + getShape(numFilter, numChannels, kernel_h, kernel_w);
		dmlScript.append(FillerUtils.getFiller(weightVar, shape, convParam.getWeightFiller(), 0));
		
		if(convParam.hasBiasFiller()) {
			ArrayList<String> bias_shape = new ArrayList<String>();
			bias_shape.add("" + numFilter);
			bias_shape.add("1");
			nColOneVar = toInt(smult(layerVar("P"), layerVar("Q"))); 
			dmlScript.append(FillerUtils.getFiller(biasVar, bias_shape, convParam.getBiasFiller(), 0));
			dmlScript.append(assign(oneVar, matrix(1, 1, nColOneVar)));
		}
		
		if(Barista.USE_MOMENTUM) {
			dmlScript.append(assign(updatePrefix+weightVar, matrix(0, nrow(weightVar), ncol(weightVar))));
			dmlScript.append(assign(updatePrefix+biasVar, matrix(0, nrow(biasVar), ncol(biasVar))));
		}
	}

	public void generateForwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		convParamStr = getInputShape() + ", " + filterShape 
				+ ", padding=[" + pad_h + "," + pad_w + "], "
				+ "stride=[" + stride_h + "," + stride_w + "]";
		 
		printForwardHeader(dmlScript);
		
		assertTop(1);
		dmlScript.append(assign(getTopDMLVar(), "conv2d(" + getBottomDMLVar() + ", " + weightVar + ", " + convParamStr + ")"));
		
		if(useBias) {
			String biasMatrix = matrix(matmult(biasVar, oneVar), "1", toInt(smult("" + numFilter, nColOneVar)));
			dmlScript.append(assign(getTopDMLVar(), add(getTopDMLVar(), biasMatrix)));
		}
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
		printBackwardHeader(dmlScript);
		assertBottom(1);
		
		if(useBias) {
			dmlScript.append(assign(gradientPrefix + biasVar, 
					rowSums(matrix(colSums(deltaVar), ""+numFilter, mult(layerVar("P"), layerVar("Q"))))));
		}
		dmlScript.append(assign(gradientPrefix + weightVar,
				"conv2d_backward_filter("+ getBottomDMLVar() + ", " + deltaVar +  ", " + convParamStr +")"));
		dmlScript.append(assign(getBottomLayerDeltaVar(),
				"conv2d_backward_data("+ weightVar + ", " + deltaVar +  ", " + convParamStr +")"));
	}

	@Override
	public String generateFinalizeDML() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		String N = getBottomLayerOutputShape(0);
		
		output_shape.clear();
		output_shape.add("" + N); // N 
		output_shape.add("" + numFilter);
		output_shape.add("P_" + layerID);
		output_shape.add("Q_" + layerID);
	}

}
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
package org.apache.sysml.api.dl.layers;

import org.apache.sysml.api.dl.DLUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import caffe.Caffe.ConvolutionParameter;
import caffe.Caffe.LayerParameter;


public class ConvolutionLayer extends Layer {

	int pad_h = 0; int pad_w = 0; int stride_h = 1; int stride_w = 1;
	int kernel_h = -1; int kernel_w = -1; int numFilter = -1;
	ConvolutionParameter convParam;
	String filterVar;
	
	public ConvolutionLayer(LayerParameter param) {
		super(param, "convOut_" + (Layer.id++));
		filterVar = "w_" + (Layer.id++);
		convParam = param.getConvolutionParam();
		
		if(convParam.hasBiasTerm())
			System.out.println("WARNING: Ignoring bias term in convolution");
		if(convParam.hasGroup())
			System.out.println("WARNING: Ignoring group in convolution");
		
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

	@Override
	public String getSetupDML() throws DMLRuntimeException {

		String warn = "";
		if(convParam.hasWeightFiller()) {
			warn = "WARN: Ignoring weight filler in convolution in initial version";
			System.out.println(warn);
		}
		checkInput();
		String numChannels = getBottomLayerOutputShape(1);
		String fShape = "shape=[" + numFilter + "," + numChannels + "," + kernel_h + "," + kernel_w + "]";
		filterShape = "filter_" + fShape;
		if(warn.equals(""))
			return filterVar + " = tensor(0, " + fShape + ")";
		else
			return filterVar + " = tensor(0, " + fShape + ") # " + warn;
	}

	public String getForwardDML() {
		return outputVar + " = conv2d(" + bottom.get(0).outputVar + ", " + filterVar + ", " + getInputShape()
				+ ", " + filterShape + ", padding=[" + pad_h + "," + pad_w + "], "
				+ "stride=[" + stride_h + "," + stride_w + "])";
	}

	@Override
	public String getBackwardDML() {
		return null;
	}

	@Override
	public String getFinalizeDML() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		String N = getBottomLayerOutputShape(0);
		String H_str = getBottomLayerOutputShape(2);
		String W_str = getBottomLayerOutputShape(3);
		
		output_shape.clear();
		output_shape.add("" + N); // N 
		output_shape.add("" + numFilter);
		output_shape.add(DLUtils.getP_DML(H_str, pad_h, kernel_h, stride_h));
		output_shape.add(DLUtils.getQ_DML(W_str, pad_w, kernel_w, stride_w));
	}

}

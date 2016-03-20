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
package org.apache.sysml.api.dl;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.sysml.api.dl.layers.AccuracyLayer;
import org.apache.sysml.api.dl.layers.ConvolutionLayer;
import org.apache.sysml.api.dl.layers.DataLayer;
import org.apache.sysml.api.dl.layers.InnerProductLayer;
import org.apache.sysml.api.dl.layers.Layer;
import org.apache.sysml.api.dl.layers.PoolingLayer;
import org.apache.sysml.api.dl.layers.ReLULayer;
import org.apache.sysml.api.dl.layers.SoftmaxWithLossLayer;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.ConvolutionUtils;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import com.google.protobuf.TextFormat;

public class DLUtils {

	public static NetParameter readCaffeNet(SolverParameter solver) throws IOException {
		InputStreamReader reader = new InputStreamReader(new FileInputStream(new File(solver.getNet())), "ASCII");
		NetParameter.Builder builder =  NetParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}

	public static SolverParameter readCaffeSolver(String fileName) throws IOException {
		InputStreamReader reader = new InputStreamReader(new FileInputStream(new File(fileName)), "ASCII");
		SolverParameter.Builder builder =  SolverParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}

	public static Layer readLayer(LayerParameter param) throws DMLRuntimeException {
		if(param.getType().equals("Data")) {
			return new DataLayer(param);
		}
		else if(param.getType().equals("Convolution")) {
			return new ConvolutionLayer(param);
		}
		else if(param.getType().equals("Pooling")) {
			return new PoolingLayer(param);
		}
		else if(param.getType().equals("InnerProduct")) {
			return new InnerProductLayer(param);
		}
		else if(param.getType().equals("ReLU")) {
			return new ReLULayer(param);
		}
		else if(param.getType().equals("Accuracy")) {
			return new AccuracyLayer(param);
		}
		else if(param.getType().equals("SoftmaxWithLoss")) {
			return new SoftmaxWithLossLayer(param);
		}
		throw new DMLRuntimeException("A layer of type " + param.getType() + " is not implemented.");
	}

	public static ArrayList<Layer> topologicalSort(ArrayList<Layer> dmlNet) {
		// TODO: 
		return dmlNet;
	}
	
	public static void setupTopBottomLayers(ArrayList<Layer> dmlNet) {
		for(Layer current : dmlNet) {
			for(int i = 0; i < current.param.getBottomCount(); i++) {
				String bottom = current.param.getBottom(i);
				if(bottom.equals("data")) {
					for(Layer l : dmlNet) {
						if(l.param.getTopCount() > 0 && l.param.getTop(0).equals(bottom)) {
							current.bottom.add(l);
						}
					}
				}
				else {
					for(Layer l : dmlNet) {
						if(l.param.getName().equals(bottom)) {
							current.bottom.add(l);
						}
					}
				}
			}
		}
	}

	public static String getP_DML(String H_str, int pad_h, int R, int stride_h) {
		String P = "((" + H_str + " + " + (2*pad_h  - R) + ") / " + stride_h + " + 1)";
		
		try {
			int H = Integer.parseInt(H_str);
			P = "" + ConvolutionUtils.getP(H, R, stride_h, pad_h);
		} catch(NumberFormatException e) {}
		return P;
	}
	
	public static String getQ_DML(String W_str, int pad_w, int S, int stride_w) {
		String Q = "((" + W_str + " + " + (2*pad_w  - S) + ") / " + stride_w + " + 1)";
		try {
			int W = Integer.parseInt(W_str);
			Q = "" + ConvolutionUtils.getP(W, S, stride_w, pad_w);
		} catch(NumberFormatException e) {}
		
		return Q;
	}
	
//	
//	public static String getP_DML(String H, String pad_h, String R, String stride_h) {
//		return "((" + H + " + 2 * " + pad_h + " - " + R + ") / " + stride_h + " + 1)";
//	}
//	
//	public static String getQ_DML(String W, String pad_w, String S, String stride_w) {
//		return "((" + W + " + 2 * " + pad_w + " - " + S + ") / " + stride_w + " + 1)";
//	}
}

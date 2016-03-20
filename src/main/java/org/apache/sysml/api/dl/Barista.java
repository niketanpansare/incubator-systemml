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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.dl.layers.Layer;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.Phase;
import caffe.Caffe.SolverParameter;

/**
 * Simple wrapper to quick try out Caffe mode. 
 * This wrapper is subject to change and is not guaranteed to be supported.
 */
public class Barista {
	public static int batchSize;
	public static int numChannelsOfInputData;
	public static int inputHeight;
	public static int inputWidth;
	
	private static void printUsage() throws DMLRuntimeException {
		throw new DMLRuntimeException("Usage is " + Barista.class.getCanonicalName() + " train -solver solver.proto numChannelsOfInputData inputHeight inputWidth");
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException, DMLException, ParseException {
		if(args.length != 6) 
			printUsage();
		
		if(args[0].equals("train") && args[1].equals("-solver")) {
			numChannelsOfInputData = Integer.parseInt(args[3]);
			inputHeight = Integer.parseInt(args[4]);
			inputWidth = Integer.parseInt(args[5]);
			
			SolverParameter solver = DLUtils.readCaffeSolver(args[2]);
			NetParameter net = DLUtils.readCaffeNet(solver);
			ArrayList<Layer> dmlNet = new ArrayList<Layer>();
			for(LayerParameter param : net.getLayerList()) {
				if(isTrainLayer(param)) {
					dmlNet.add(DLUtils.readLayer(param));
				}
			}
			DLUtils.setupTopBottomLayers(dmlNet);
			dmlNet = DLUtils.topologicalSort(dmlNet);
			String script = (new ForwardBackwardSolver()).getForwardBackwardDML(dmlNet, solver);
			
			// Current version needs Spark installation. Will remove this requirement in later version
			// MLContext ml = new MLContext(new SparkContext());
			
			System.out.println("\n-----------------------------------\n");
			System.out.println(script);
			System.out.println("\n-----------------------------------\n");
			//ml.executeScript(script);
		}
		else {
			printUsage();
		}
	}
	
	private static boolean isTrainLayer(LayerParameter param) {
		if(param.getIncludeCount() > 0 && param.getInclude(0).hasPhase() && param.getInclude(0).getPhase() == Phase.TEST) {
			return false;
		}
		return true;
	}
}

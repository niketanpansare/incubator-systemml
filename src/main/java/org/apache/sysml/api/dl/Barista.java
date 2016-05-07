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
import org.apache.sysml.api.dl.layer.Layer;
import org.apache.sysml.api.dl.solver.ForwardBackwardSolver;
import org.apache.sysml.api.dl.utils.DLUtils;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.Phase;
import caffe.Caffe.SolverParameter;

/**
 * Simple wrapper to quick try out Caffe mode. 
 * This wrapper is subject to change and is not guaranteed to be supported.
 * 
 * org.apache.sysml.api.dl.Barista train -solver LenetSolver.proto
 */
public class Barista {
	public static int batchSize;
	
	// Constants variable names
	public static String numChannelsOfInputData = "numChannels";
	public static String inputHeight = "img_height";
	public static String inputWidth = "img_width";
	public static String numClasses = "numClasses";
	public static String numValidation = "numValidation";
	
	private static void printUsage() throws DMLRuntimeException {
		throw new DMLRuntimeException("Usage is " + Barista.class.getCanonicalName() + " train -solver solver.proto");
	}
	
	// TODO: remove this main method and only allow calls to Barista through DMLScript
	public static void main(String[] args) throws FileNotFoundException, IOException, DMLException, ParseException {
		if(args.length != 3) 
			printUsage();
		
		if(args[0].equals("train") && args[1].equals("-solver")) {
			String script = getTrainingDMLScript(args[2]);
			System.out.println("\n-----------------------------------\n");
			System.out.println(script);
			System.out.println("\n-----------------------------------\n");
		}
		else {
			printUsage();
		}
	}
	
	// TODO: This can be called directly by DMLScript when -caffe=train flag is turned on 
	public static String getTrainingDMLScript(String solverFilePath) throws IOException, DMLRuntimeException, LanguageException {
		SolverParameter solver = DLUtils.readCaffeSolver(solverFilePath);
		NetParameter net = DLUtils.readCaffeNet(solver);
		ArrayList<Layer> dmlNet = new ArrayList<Layer>();
		for(LayerParameter param : net.getLayerList()) {
			if(isTrainLayer(param)) {
				dmlNet.add(DLUtils.readLayer(param));
			}
		}
		DLUtils.setupTopBottomLayers(dmlNet);
		dmlNet = DLUtils.topologicalSort(dmlNet);
		return (new ForwardBackwardSolver()).getForwardBackwardDML(dmlNet, solver);
	}
	
	private static boolean isTrainLayer(LayerParameter param) {
		if(param.getIncludeCount() > 0 && param.getInclude(0).hasPhase() && param.getInclude(0).getPhase() == Phase.TEST) {
			return false;
		}
		return true;
	}
}
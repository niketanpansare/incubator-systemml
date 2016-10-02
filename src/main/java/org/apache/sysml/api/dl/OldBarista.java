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
import java.util.HashMap;

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
public class OldBarista {
	
	public static enum DMLPhase { TRAIN, TEST, VALIDATE };
	public static DMLPhase currentPhase = DMLPhase.TRAIN;
	
	public static int BATCH_SIZE;
	public static boolean USE_MOMENTUM = false;
	public static ArrayList<Layer> dmlNet;
	// Matches the top to DML variable of each layer
	public static HashMap<String, ArrayList<String>> blobToDMLVariableMapping = new HashMap<String, ArrayList<String>>();
	// Matches the bottom to DML variable of each layer
	public static HashMap<String, ArrayList<String>> blobToDeltaVariableMapping = new HashMap<String, ArrayList<String>>();
	
	// Constants variable names
	public static String numChannelsOfInputData = "numChannels";
	public static String inputHeight = "img_height";
	public static String inputWidth = "img_width";
	public static String numClasses = "numClasses";
	public static String numValidation = "numValidation";
	public static String numTest = "numTest";
	public static String step_size = "step_size";
	public static String trainingVarsuffix = "b";
	public static String validationVarsuffix = "v";
	public static String testVarsuffix = "t";
	
	private static void printUsage() throws DMLRuntimeException {
		throw new DMLRuntimeException("Usage is " + OldBarista.class.getCanonicalName() + " train -solver solver.proto");
	}
	
	// TODO: remove this main method and only allow calls to Barista through DMLScript
	public static void main(String[] args) throws FileNotFoundException, IOException, DMLException, ParseException {
		if(args.length != 3) 
			printUsage();
		
		if(args[0].equals("train") && args[1].equals("-solver")) {
			String script = getTrainingDMLScript(args[2]);
			System.out.println(script);
		}
		else {
			printUsage();
		}
	}
	
	// TODO: This can be called directly by DMLScript when -caffe=train flag is turned on 
	public static String getTrainingDMLScript(String solverFilePath) throws IOException, DMLRuntimeException, LanguageException {
		SolverParameter solver = DLUtils.readCaffeSolver(solverFilePath);
		NetParameter net = DLUtils.readCaffeNet(solver);
		net = handleInPlace(net);
		dmlNet = new ArrayList<Layer>();
		blobToDMLVariableMapping.clear();
		for(LayerParameter param : net.getLayerList()) {
			if(isTrainLayer(param))
				dmlNet.add(DLUtils.readLayer(param));
		}
		
		return (new ForwardBackwardSolver()).getForwardBackwardDML(dmlNet, solver);
	}
	
	public static NetParameter handleInPlace(NetParameter net) throws DMLRuntimeException {
		for(LayerParameter param : net.getLayerList()) {
			for(String t : param.getTopList()) {
				for(String b : param.getBottomList()) {
					if(t.equalsIgnoreCase(b)) {
						throw new DMLRuntimeException("ERROR: In-place layers is not supported: " + "\"" + param.getType() + " layer:" + param.getName() + "\"");
					}
				}
			}
		}
		return net;
	}
	
	public static boolean isTrainLayer(LayerParameter param) {
		if(param.getIncludeCount() > 0 && param.getInclude(0).hasPhase() && param.getInclude(0).getPhase() == Phase.TEST) {
			return false;
		}
		return true;
	}
	
}
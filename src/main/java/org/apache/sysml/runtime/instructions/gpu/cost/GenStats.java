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
package org.apache.sysml.runtime.instructions.gpu.cost;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;

/**
 * Generates statistics for matrix multiplication
 */
public class GenStats {
	// Parameters to adjust
	private static int [] dimensions = { 1, 5, 10, 50, 100, 200, 500, 1000, 2000 };
	private static double [] sparsity = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99};
	private static int numCallsPerOp = 10; // This helps to reduce ms
	private static int numIter = 3;
	
	public static void main(String[] args) throws DMLRuntimeException, IOException { 
		initGPU();
		matrixMultiplication();
	}
	
	private static void initGPU() {
		DMLScript.USE_ACCELERATOR = true;
		GPUContext.createGPUContext(); // Set GPU memory budget
	}
	
	/**
	 * Generates a random matrix of given sparsity
	 * @param rows
	 * @param cols
	 * @param sparsity
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock generateRandomMatrixBlock(int rows, int cols, double sparsity) throws DMLRuntimeException {
		int seed = 12345;
//		RandomMatrixGenerator gen = LibMatrixDatagen.createRandomMatrixGenerator("uniform", rows, cols, 
//				defaultBlockSize(), defaultBlockSize(), sparsity, 0, 1);
		RandomMatrixGenerator gen = new RandomMatrixGenerator("uniform", rows, cols, 
				ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(), sparsity, 0, 1);
		MatrixBlock mb = MatrixBlock.randOperations(gen, seed);
		return mb;
	}
	
	static int maxBlockSize = -1;
	/**
	 * Return the default matrix block size.
	 * 
	 * @return the default matrix block size
	 */
	public static int defaultBlockSize() {
//		if(maxBlockSize == -1) {
//			for(int i = 0; i < dimensions.length; i++)
//				maxBlockSize = Math.max(dimensions[i], maxBlockSize);
//		}
//		return maxBlockSize;
		return ConfigurationManager.getBlocksize();
	}

	/**
	 * Return the location of the scratch space directory.
	 * 
	 * @return the location of the scratch space directory
	 */
	public static String scratchSpace() {
		return ConfigurationManager.getScratchSpace(); 
	}
	
	/**
	 * Returns a matrix object that encapsulates the given matrix object.
	 * @param mb
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String getMatrixObject(ExecutionContext ec, MatrixBlock mb) throws DMLRuntimeException {
		MatrixCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), defaultBlockSize(), defaultBlockSize(), mb.getNonZeros());
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		String outVarName = "matVar_" + (_varID++);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, scratchSpace() + "/" + outVarName, meta);
		mo.acquireModify(mb);
		mo.release();
		ec.setVariable(outVarName, mo);
		ec.allocateGPUMatrixObject(outVarName);
		return outVarName;
	}
	
	/**
	 * Utility to write to a text file
	 * @param file
	 * @param text
	 * @throws IOException
	 */
	private static void appendText(String file, String text) throws IOException {
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(file, true)));
	    out.println(text);
	    out.close();
	}
	
	/**
	 * Creates an output matrix object and registers it to the given execution context.
	 * 
	 * @param ec
	 * @param rows
	 * @param cols
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static String createOutputMatrixObject(ExecutionContext ec, int rows, int cols) throws DMLRuntimeException {
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, defaultBlockSize(), defaultBlockSize());
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		String outVarName = "matVar_" + (_varID++);
		MatrixObject mo = new MatrixObject(ValueType.DOUBLE, scratchSpace() + "/" + outVarName, meta);
		ec.setVariable(outVarName, mo);
		return outVarName;
	}
	
	static MatrixObject mo1 =  null;
	static MatrixObject mo2 = null;
	private static void cpToGPUInputs(ExecutionContext ec, String input1, String input2) throws DMLRuntimeException {
		mo1 = ec.getMatrixInputForGPUInstruction(input1);
        mo2 = ec.getMatrixInputForGPUInstruction(input2);
	}
	
	private static void gpuToCP(ExecutionContext ec, String input1, String input2, String outVar) throws DMLRuntimeException {
		//release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(input1);
		ec.releaseMatrixInputForGPUInstruction(input2);
		ec.releaseMatrixOutputForGPUInstruction(outVar);
		
		ec.getMatrixObject(input1).getGPUObject().acquireHostModify(); // Ensures that GPU data is deleted
		ec.getMatrixObject(input2).getGPUObject().acquireHostModify(); // Ensures that GPU data is deleted
		
		ec.getMatrixInput(outVar); // Pull output to CP so that matmult is executed
		ec.releaseMatrixInput(outVar);
		ec.getMatrixObject(outVar).getGPUObject().acquireHostModify(); // Ensures that GPU data is deleted	
	}
	
	private static void gpuToCP(ExecutionContext ec, String input1, String input2) throws DMLRuntimeException {
		//release inputs/outputs
		ec.releaseMatrixInputForGPUInstruction(input1);
		ec.releaseMatrixInputForGPUInstruction(input2);
		ec.getMatrixObject(input1).getGPUObject().acquireHostModify(); // Ensures that GPU data is deleted
		ec.getMatrixObject(input2).getGPUObject().acquireHostModify(); // Ensures that GPU data is deleted
	}
	
	/**
	 * Run matrix multiplication for different scenarios.
	 * 
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private static void matrixMultiplication() throws DMLRuntimeException, IOException {
		int numThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		ExecutionContext ec = ExecutionContextFactory.createContext(null);
		String statsFile = "matMultStats.csv";
		String costFile = "matMultStats.csv";
		for(int it = 0; it < numIter; it++) {
			for(int s1 = 0; s1 < sparsity.length; s1++) {
				for(int s2 = 0; s2 < sparsity.length; s2++) {
					for(int i = 0; i < dimensions.length; i++) {
						for(int j = 0; j < dimensions.length; j++) {
							for(int k = 0; k < dimensions.length; k++) {
								int dim1 = dimensions[i];
								int dim2 = dimensions[j];
								int dim3 = dimensions[k];
								MatrixBlock mb1 = generateRandomMatrixBlock(dim1, dim2, sparsity[s1]);
								MatrixBlock mb2 = generateRandomMatrixBlock(dim2, dim3, sparsity[s2]);
								
								String metaData = "" + dim1 + "," + dim2 + "," + dim3 + "," + sparsity[s1] + "," + sparsity[s2] + ",";
								
								// -------------------------------------------
								// Setup 1
								try {
									long t1 = System.nanoTime();
									for(int iter = 0; iter < numCallsPerOp; iter++) {
										LibMatrixMult.matrixMult(mb1, mb2, new MatrixBlock(dim1, dim3, -1) , numThreads);
									}
									appendText(statsFile, "CP_exec," + metaData + ((System.nanoTime() - t1)*1e-6));
								} catch(Exception e) {
									appendText(statsFile, "CP_exec," + metaData + "-1");
								}
								// -------------------------------------------
								
								String input1 = getMatrixObject(ec, mb1);
								String input2 = getMatrixObject(ec, mb2);
								
								String outVar = createOutputMatrixObject(ec, dim1, dim3);
								
								// -------------------------------------------
								// Setup 2
								try {
									long t2 = System.nanoTime();
									cpToGPUInputs(ec, input1, input2);
									for(int iter = 0; iter < numCallsPerOp; iter++) {
										LibMatrixCUDA.matmult(ec, mo1, mo2, outVar, false, false);
									}
									gpuToCP(ec, input1, input2, outVar);
									// Inputs and Output on GPU 
									appendText(statsFile, "GPU_exec_GGG," + metaData + ((System.nanoTime() - t2)*1e-6));
								} catch(Exception e) {
									// To handle cases such as 
//									org.apache.sysml.runtime.DMLRuntimeException: There is not enough memory on device for this matrix!
//							        at org.apache.sysml.runtime.instructions.gpu.context.GPUObject.evict(GPUObject.java:169)
									appendText(statsFile, "GPU_exec_GGG," + metaData + "-1");
								}
								// -------------------------------------------
								
								// -------------------------------------------
								// Setup 3
								try {
									long t3 = System.nanoTime();
									for(int iter = 0; iter < numCallsPerOp; iter++) {
										cpToGPUInputs(ec, input1, input2);
										LibMatrixCUDA.matmult(ec, mo1, mo2, outVar, false, false);
										gpuToCP(ec, input1, input2, outVar);
									}
									// Inputs and Output on CP
									appendText(statsFile, "GPU_exec_CCC," + metaData + ((System.nanoTime() - t3)*1e-6));
								} catch(Exception e) {
									appendText(statsFile, "GPU_exec_CCC," + metaData + "-1");
								}
								// -------------------------------------------
								
								// -------------------------------------------
								// Setup 4
								try {
									long t4 = System.nanoTime();
									for(int iter = 0; iter < numCallsPerOp; iter++) {
										cpToGPUInputs(ec, input1, input2);
										LibMatrixCUDA.matmult(ec, mo1, mo2, outVar, false, false);
										gpuToCP(ec, input1, input2);
									}
									// Inputs on CP and Outputs on GPU
									appendText(statsFile, "GPU_exec_CCG," + metaData + ((System.nanoTime() - t4)*1e-6));
								} catch(Exception e) {
									appendText(statsFile, "GPU_exec_CCG," + metaData + "-1");
								}
								// -------------------------------------------
								
							}
						}
					}
				}
			}
		}
	}
	
	private static long _varID = 1; 
}

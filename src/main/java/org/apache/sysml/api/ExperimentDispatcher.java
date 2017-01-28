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
package org.apache.sysml.api;

import java.io.BufferedWriter;
import java.io.FileWriter;

import org.apache.sysml.lops.MMTSJ.MMTSJType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;

public class ExperimentDispatcher {
	
	private static MatrixBlock m1 = null;
	private static MatrixBlock m2 = null;
	private static int M = 1000000;
	private static int K = 1000;
	private static int NUM_ITERS = 20;
	public static void runExperiment(int num, String[] args) throws DMLRuntimeException {
		switch(num){
			case 1:
				// Matrix-vector 1M x 1K, dense (~8GB)
				runExperiment(num, M, K, 1, 0.9);
				return;
			case 2:
				// Vector-matrix 1M x 1K, dense (~8GB)
				runExperiment(num, 1, M, K, 0.9);
				return;	
//			case 3:
//				// Matrix-vector/vector-matrix 1M x 10K, sparsity=0.1 (~12GB)
//				runExperiment(num, M, 10*K, 1, 0.1);
//				return;
//			case 4:
//				// Vector-matrix 1M x 10K, sparsity=0.1 (~12GB)
//				runExperiment(num, 1, M, 10*K, 0.1);
//				return;
			case 5:
				// Matrix-matrix/matrix-Matrix 1M x 1K x 20, dense (40GFLOPs)
				runExperiment(num, M, K, 20, 0.9);
				return;
//			case 6:
//				// Matrix-matrix/matrix-Matrix 1M x 10K (sparsity=0.1) x 20, (40 GFLOPs)
//				runExperiment(num, M, 10*K, 20, 0.1);
//				return;
			case 7:
				// Squared matrix 3K x 3K, dense (54 GFLOPs)
				runExperiment(num, 3*K, 3*K, 3*K, 0.9);
				return;
			case 8:
				// n rows x 196,608 columns %*% 196,608 x 512
				runExperiment(num, 64, 196608, 512, 0.9);
				return;
			case 9:
				// n rows x 196,608 columns %*% 196,608 x 512
				runExperiment(num, 1024, 196608, 512, 0.9);
				return;
			case 10:
				runTSMMExperiment(num, M, K, true, 0.9);
				return;
			case 11:
				runTSMMExperiment(num, M, K, false, 0.9);
				return;
			default:
				throw new RuntimeException("ExperimentDispatcher: unable to find experiment: "+num);
		}
		
	}
	
	private static void runTSMMExperiment(int num, int m, int n, boolean isLeft, double sparsity) throws DMLRuntimeException {
		try {
			m1 = MatrixBlock.randOperations(m, n, sparsity, 0, 1, "uniform", 1234);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("ExperimentDispatcher: unable to run experiment", e);
		}
		
		double results[][] = new double[NUM_ITERS][4]; //single/multi-threaded
		int k = InfrastructureAnalyzer.getLocalParallelism();
		for(int i = -5; i < NUM_ITERS; i++) {
			Timing time1 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = false;
			m1.transposeSelfMatrixMultOperations(new MatrixBlock(), isLeft ? MMTSJType.LEFT : MMTSJType.RIGHT, 1 );
			double t1 = time1.stop();
			
			Timing time2 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = true;
			m1.transposeSelfMatrixMultOperations(new MatrixBlock(), isLeft ? MMTSJType.LEFT : MMTSJType.RIGHT, 1 );
			double t2 = time2.stop();
			
			Timing time3 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = false;
			m1.transposeSelfMatrixMultOperations(new MatrixBlock(), isLeft ? MMTSJType.LEFT : MMTSJType.RIGHT, k );
			double t3 = time3.stop();
			
			Timing time4 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = true;
			m1.transposeSelfMatrixMultOperations(new MatrixBlock(), isLeft ? MMTSJType.LEFT : MMTSJType.RIGHT, k );
			double t4 = time4.stop();
			
			if(i >= 0) {
				results[i][0] = t1;
				results[i][1] = t2;
				results[i][2] = t3;
				results[i][3] = t4;
			}
		}
		writeResultFile("result_" + num +".txt", results);
	}
	
	private static void runExperiment(int num, int m, int n, int k, double sparsity) throws DMLRuntimeException {
		generateInputs(m, n, k, sparsity);
		double results[][] = new double[NUM_ITERS][4]; //single/multi-threaded
		for(int i = -5; i < NUM_ITERS; i++) {
			Timing time1 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = false;
			m1.aggregateBinaryOperations(m1, m2, new MatrixBlock(), getMatMultOperator(true));
			double t1 = time1.stop();
			
			Timing time2 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = true;
			m1.aggregateBinaryOperations(m1, m2, new MatrixBlock(), getMatMultOperator(true));
			double t2 = time2.stop();
			
			Timing time3 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = false;
			m1.aggregateBinaryOperations(m1, m2, new MatrixBlock(), getMatMultOperator(false));
			double t3 = time3.stop();
			
			Timing time4 = new Timing(true);
			DMLScript.ENABLE_NATIVE_BLAS = true;
			m1.aggregateBinaryOperations(m1, m2, new MatrixBlock(), getMatMultOperator(false));
			double t4 = time4.stop();
			
			if(i >= 0) {
				results[i][0] = t1;
				results[i][1] = t2;
				results[i][2] = t3;
				results[i][3] = t4;
			}
		}
		writeResultFile("result_" + num +".txt", results);
	}
	
	private static AggregateBinaryOperator getMatMultOperator(boolean isSingleThreaded) {
		if(isSingleThreaded) {
			AggregateOperator aop1 = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop1 = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop1);
			return abop1;
		}
		else {
			AggregateOperator aop1 = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop1 = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop1, InfrastructureAnalyzer.getLocalParallelism());
			return abop1;
		}
	}
	
	private static void generateInputs(int m, int n, int k, double sparsity) {
		generateInputs(m, n, k, sparsity, 0.9);
	}
	private static void generateInputs(int m, int n, int k, double sparsity1, double sparsity2) {
		try {
			m1 = MatrixBlock.randOperations(m, n, sparsity1, 0, 1, "uniform", 1234);
			m2 = MatrixBlock.randOperations(n, k, sparsity2, 0, 1, "uniform", 5467);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("ExperimentDispatcher: unable to run experiment", e);
		}
	}
	
	protected static void writeResultFile(String fname, double[][] results)
	{
		try
		{
			FileWriter fw = new FileWriter(fname);
			BufferedWriter bw = new BufferedWriter(fw);
			
			for( int i=0; i<results.length; i++ ) {
				bw.write(String.valueOf(results[i][0]));
				for( int j=1; j<results[i].length; j++ ) {
					bw.write("\t");
					bw.write(String.valueOf(results[i][j]));
				}
				bw.write("\n");
			}
			
			bw.close();
			fw.close();
		}
		catch(Exception ex){
			throw new RuntimeException(ex);
		}
	}
}
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

package org.apache.sysml.test.gpu;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.runtime.instructions.gpu.DnnGPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.DnnGPUInstruction.LstmOperator;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * Tests lstm builtin function
 */
public class LstmTest extends GPUTests {

	private final static String TEST_NAME = "LstmTests";
	private final int seed = 42;

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testLstmForward1() {
		testLstmCuDNNWithNN(1, 1, 1, 1, "TRUE");
	}
	
	@Test
	public void testLstmForward2() {
		testLstmCuDNNWithNN(1, 1, 1, 1, "FALSE");
	}
	
	@Test
	public void testLstmForward3() {
		testLstmCuDNNWithNN(2, 3, 5, 10, "TRUE");
	}
	
	@Test
	public void testLstmForward4() {
		testLstmCuDNNWithNN(2, 3, 5, 10, "FALSE");
	}
	
	@Test
	public void testLstmForward5() {
		testLstmCuDNNWithNN(1, 3, 5, 1, "TRUE");
	}
	
	@Test
	public void testLstmForward6() {
		testLstmCuDNNWithNN(1, 3, 5, 1, "FALSE");
	}
	
	public void testLstmCuDNNWithNN(int N, int T, int D, int M, String returnSequences) {
		String scriptStr = "source(\"nn/layers/lstm_staging.dml\") as lstm;\n "
				+ "[output, c] = lstm::forward(x, w, b, " + returnSequences + ", out0, c0)";
		
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateNonRandomInputMatrix(spark, N, T*D));
		inputs.put("w", generateNonRandomInputMatrix(spark, D+M, 4*M));
		inputs.put("b", generateNonRandomInputMatrix(spark, 1, 4*M));
		inputs.put("out0", generateNonRandomInputMatrix(spark, N, M));
		inputs.put("c0", generateNonRandomInputMatrix(spark, N, M));
		List<String> outputs = Arrays.asList("output", "c");
		List<Object> outGPUWithCuDNN = null;
		List<Object> outGPUWithDNN = null;
		synchronized (DnnGPUInstruction.FORCED_LSTM_OP) {
			DnnGPUInstruction.FORCED_LSTM_OP = LstmOperator.CUDNN;
			outGPUWithCuDNN = runOnGPU(spark, scriptStr, inputs, outputs);
			inputs = new HashMap<>();
			inputs.put("x", generateNonRandomInputMatrix(spark, N, T*D));
			inputs.put("w", generateNonRandomInputMatrix(spark, D+M, 4*M));
			inputs.put("b", generateNonRandomInputMatrix(spark, 1, 4*M));
			inputs.put("out0", generateNonRandomInputMatrix(spark, N, M));
			inputs.put("c0", generateNonRandomInputMatrix(spark, N, M));
			DnnGPUInstruction.FORCED_LSTM_OP = LstmOperator.DENSE_NN;
			outGPUWithDNN = runOnGPU(spark, scriptStr, inputs, outputs);
		}
		assertEqualObjects(outGPUWithCuDNN.get(0), outGPUWithDNN.get(0));
		assertEqualObjects(outGPUWithCuDNN.get(1), outGPUWithDNN.get(1));
	}
}

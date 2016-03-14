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

package org.apache.sysml.test.integration.functions.tensor;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.DMLScript.TensorLayout;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



public class RightIndexingMatrixTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "RightIndexingMatrixTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RightIndexingMatrixTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B", "C", "D"}));
	}
	
	@Test
	public void testRightIndexingDenseCPW_XYZ() 
	{
		runRightIndexingTest(ExecType.CP, TensorLayout.W_XYZ);
	}
	
	@Test
	public void testRightIndexingDenseCPWXY_Z() 
	{
		runRightIndexingTest(ExecType.CP, TensorLayout.WXY_Z);
	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runRightIndexingTest( ExecType et, TensorLayout layout) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
			
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		TensorLayout layoutOld = DMLScript.tensorLayout;
		DMLScript.tensorLayout = layout;
		
		try
		{
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
		    else {
		    	rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
		    }
			if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			loadTestConfiguration(config);
	        
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			
			programArgs = new String[]{"-args",  
					output("B"), output("C"), output("D") };
			        
			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			// A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((3, 2, 2))
			// B = A[0:2, :, :]
			// C = A[1:, :,:][0:1, :,:]
			
			HashMap<CellIndex, Double> bHM = new HashMap<CellIndex, Double>();
			if(DMLScript.tensorLayout == TensorLayout.WXY_Z) {
				bHM.put(new CellIndex(1, 1), 1.0); bHM.put(new CellIndex(1, 2), 2.0);
				bHM.put(new CellIndex(2, 1), 3.0); bHM.put(new CellIndex(2, 2), 4.0);
				bHM.put(new CellIndex(3, 1), 5.0); bHM.put(new CellIndex(3, 2), 6.0);
				bHM.put(new CellIndex(4, 1), 7.0); bHM.put(new CellIndex(4, 2), 8.0);
			}
			else if(DMLScript.tensorLayout == TensorLayout.W_XYZ) {
				bHM.put(new CellIndex(2, 2), 6.0); bHM.put(new CellIndex(1, 1), 1.0);
				bHM.put(new CellIndex(2, 3), 7.0); bHM.put(new CellIndex(1, 3), 3.0);
				bHM.put(new CellIndex(2, 1), 5.0); bHM.put(new CellIndex(1, 2), 2.0);
				bHM.put(new CellIndex(1, 4), 4.0); bHM.put(new CellIndex(2, 4), 8.0);
			}
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "NumPy");
			
			HashMap<CellIndex, Double> cHM = new HashMap<CellIndex, Double>();
			if(DMLScript.tensorLayout == TensorLayout.WXY_Z) {
				cHM.put(new CellIndex(1, 1), 5.0); cHM.put(new CellIndex(1, 2), 6.0);
				cHM.put(new CellIndex(2, 1), 7.0); cHM.put(new CellIndex(2, 2), 8.0);
			}
			else if(DMLScript.tensorLayout == TensorLayout.W_XYZ) {
				cHM.put(new CellIndex(1, 1), 5.0); cHM.put(new CellIndex(1, 3), 7.0);
				cHM.put(new CellIndex(1, 2), 6.0); cHM.put(new CellIndex(1, 4), 8.0);
			}
			dmlfile = readDMLMatrixFromHDFS("C");
			TestUtils.compareMatrices(dmlfile, cHM, epsilon, "C-DML", "NumPy");
			
		}
		finally
		{
			DMLScript.tensorLayout = layoutOld;
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}

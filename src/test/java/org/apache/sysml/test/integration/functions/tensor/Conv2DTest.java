package org.apache.sysml.test.integration.functions.tensor;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.api.DMLScript.TensorLayout;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class Conv2DTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "Conv2DTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + Conv2DTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B"}));
	}
	
	HashMap<CellIndex, Double> bHM = new HashMap<CellIndex, Double>();
	
	@Test
	public void testConv2DDense1() 
	{
		int imgSize = 3; int numChannels = 3; int numFilters = 6; int filterSize = 2; int stride = 1; int pad = 0;
		
		bHM.clear();
		bHM.put(new CellIndex(1, 1), 1245.0); bHM.put(new CellIndex(1, 2), 1323.0); bHM.put(new CellIndex(1, 3), 1479.0); bHM.put(new CellIndex(1, 4), 1557.0); 
		bHM.put(new CellIndex(1, 5), 2973.0); bHM.put(new CellIndex(1, 6), 3195.0); bHM.put(new CellIndex(1, 7), 3639.0); bHM.put(new CellIndex(1, 8), 3861.0); 
		bHM.put(new CellIndex(1, 9), 4701.0); bHM.put(new CellIndex(1, 10), 5067.0); bHM.put(new CellIndex(1, 11), 5799.0); bHM.put(new CellIndex(1, 12), 6165.0); 
		bHM.put(new CellIndex(1, 13), 6429.0); bHM.put(new CellIndex(1, 14), 6939.0); bHM.put(new CellIndex(1, 15), 7959.0); bHM.put(new CellIndex(1, 16), 8469.0); 
		bHM.put(new CellIndex(1, 17), 8157.0); bHM.put(new CellIndex(1, 18), 8811.0); bHM.put(new CellIndex(1, 19), 10119.0); bHM.put(new CellIndex(1, 20), 10773.0); 
		bHM.put(new CellIndex(1, 21), 9885.0); bHM.put(new CellIndex(1, 22), 10683.0); bHM.put(new CellIndex(1, 23), 12279.0); bHM.put(new CellIndex(1, 24), 13077.0); 
		bHM.put(new CellIndex(2, 1), 3351.0); bHM.put(new CellIndex(2, 2), 3429.0); bHM.put(new CellIndex(2, 3), 3585.0); bHM.put(new CellIndex(2, 4), 3663.0); 
		bHM.put(new CellIndex(2, 5), 8967.0); bHM.put(new CellIndex(2, 6), 9189.0); bHM.put(new CellIndex(2, 7), 9633.0); bHM.put(new CellIndex(2, 8), 9855.0); 
		bHM.put(new CellIndex(2, 9), 14583.0); bHM.put(new CellIndex(2, 10), 14949.0); bHM.put(new CellIndex(2, 11), 15681.0); bHM.put(new CellIndex(2, 12), 16047.0); 
		bHM.put(new CellIndex(2, 13), 20199.0); bHM.put(new CellIndex(2, 14), 20709.0); bHM.put(new CellIndex(2, 15), 21729.0); bHM.put(new CellIndex(2, 16), 22239.0); 
		bHM.put(new CellIndex(2, 17), 25815.0); bHM.put(new CellIndex(2, 18), 26469.0); bHM.put(new CellIndex(2, 19), 27777.0); bHM.put(new CellIndex(2, 20), 28431.0); 
		bHM.put(new CellIndex(2, 21), 31431.0); bHM.put(new CellIndex(2, 22), 32229.0); bHM.put(new CellIndex(2, 23), 33825.0); bHM.put(new CellIndex(2, 24), 34623.0); 
		bHM.put(new CellIndex(3, 1), 5457.0); bHM.put(new CellIndex(3, 2), 5535.0); bHM.put(new CellIndex(3, 3), 5691.0); bHM.put(new CellIndex(3, 4), 5769.0); 
		bHM.put(new CellIndex(3, 5), 14961.0); bHM.put(new CellIndex(3, 6), 15183.0); bHM.put(new CellIndex(3, 7), 15627.0); bHM.put(new CellIndex(3, 8), 15849.0); 
		bHM.put(new CellIndex(3, 9), 24465.0); bHM.put(new CellIndex(3, 10), 24831.0); bHM.put(new CellIndex(3, 11), 25563.0); bHM.put(new CellIndex(3, 12), 25929.0); 
		bHM.put(new CellIndex(3, 13), 33969.0); bHM.put(new CellIndex(3, 14), 34479.0); bHM.put(new CellIndex(3, 15), 35499.0); bHM.put(new CellIndex(3, 16), 36009.0); 
		bHM.put(new CellIndex(3, 17), 43473.0); bHM.put(new CellIndex(3, 18), 44127.0); bHM.put(new CellIndex(3, 19), 45435.0); bHM.put(new CellIndex(3, 20), 46089.0); 
		bHM.put(new CellIndex(3, 21), 52977.0); bHM.put(new CellIndex(3, 22), 53775.0); bHM.put(new CellIndex(3, 23), 55371.0); bHM.put(new CellIndex(3, 24), 56169.0); 
		bHM.put(new CellIndex(4, 1), 7563.0); bHM.put(new CellIndex(4, 2), 7641.0); bHM.put(new CellIndex(4, 3), 7797.0); bHM.put(new CellIndex(4, 4), 7875.0); 
		bHM.put(new CellIndex(4, 5), 20955.0); bHM.put(new CellIndex(4, 6), 21177.0); bHM.put(new CellIndex(4, 7), 21621.0); bHM.put(new CellIndex(4, 8), 21843.0); 
		bHM.put(new CellIndex(4, 9), 34347.0); bHM.put(new CellIndex(4, 10), 34713.0); bHM.put(new CellIndex(4, 11), 35445.0); bHM.put(new CellIndex(4, 12), 35811.0); 
		bHM.put(new CellIndex(4, 13), 47739.0); bHM.put(new CellIndex(4, 14), 48249.0); bHM.put(new CellIndex(4, 15), 49269.0); bHM.put(new CellIndex(4, 16), 49779.0); 
		bHM.put(new CellIndex(4, 17), 61131.0); bHM.put(new CellIndex(4, 18), 61785.0); bHM.put(new CellIndex(4, 19), 63093.0); bHM.put(new CellIndex(4, 20), 63747.0); 
		bHM.put(new CellIndex(4, 21), 74523.0); bHM.put(new CellIndex(4, 22), 75321.0); bHM.put(new CellIndex(4, 23), 76917.0); bHM.put(new CellIndex(4, 24), 77715.0); 
		bHM.put(new CellIndex(5, 1), 9669.0); bHM.put(new CellIndex(5, 2), 9747.0); bHM.put(new CellIndex(5, 3), 9903.0); bHM.put(new CellIndex(5, 4), 9981.0); 
		bHM.put(new CellIndex(5, 5), 26949.0); bHM.put(new CellIndex(5, 6), 27171.0); bHM.put(new CellIndex(5, 7), 27615.0); bHM.put(new CellIndex(5, 8), 27837.0); 
		bHM.put(new CellIndex(5, 9), 44229.0); bHM.put(new CellIndex(5, 10), 44595.0); bHM.put(new CellIndex(5, 11), 45327.0); bHM.put(new CellIndex(5, 12), 45693.0); 
		bHM.put(new CellIndex(5, 13), 61509.0); bHM.put(new CellIndex(5, 14), 62019.0); bHM.put(new CellIndex(5, 15), 63039.0); bHM.put(new CellIndex(5, 16), 63549.0); 
		bHM.put(new CellIndex(5, 17), 78789.0); bHM.put(new CellIndex(5, 18), 79443.0); bHM.put(new CellIndex(5, 19), 80751.0); bHM.put(new CellIndex(5, 20), 81405.0); 
		bHM.put(new CellIndex(5, 21), 96069.0); bHM.put(new CellIndex(5, 22), 96867.0); bHM.put(new CellIndex(5, 23), 98463.0); bHM.put(new CellIndex(5, 24), 99261.0); 
		
		runConv2DTest(ExecType.CP, TensorLayout.W_XYZ, "NCHW", imgSize, numChannels, numFilters, filterSize, stride, pad);
	}
	
	
	// TODO: Make convolution more robust
//	@Test
//	public void testConv2DDense2() 
//	{
//		int imgSize = 10; int numChannels = 3; int numFilters = 2; int filterSize = 3; int stride = 3; int pad = 1;
//		bHM.clear();
//	}
	
	/**
	 * 
	 * @param et
	 * @param sparse
	 */
	public void runConv2DTest( ExecType et, TensorLayout layout, String imgFormat, int imgSize, int numChannels, int numFilters, 
			int filterSize, int stride, int pad) 
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
			
			programArgs = new String[]{"-args",  "" + imgSize,
					imgFormat, "" + numChannels, "" + numFilters, 
					"" + filterSize, "" + stride, "" + pad, 
					output("B")};
			        
			boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			TestUtils.compareMatrices(dmlfile, bHM, epsilon, "B-DML", "NumPy");
			
		}
		finally
		{
			DMLScript.tensorLayout = layoutOld;
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}

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

package org.apache.sysml.test.integration.functions.frame;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction.AppendType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class FrameAppendTest extends AutomatedTestBase
{
	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsStringsCBind()  {
		runFrameAppendTest(schemaStrings, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsStringsRBind()  { //note: ncol(A)=ncol(B)
		runFrameAppendTest(schemaStrings, schemaStrings, AppendType.RBIND);
	}
	
	@Test
	public void testFrameMixedStringsCBind()  {
		runFrameAppendTest(schemaMixed, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsMixedCBind()  {
		runFrameAppendTest(schemaStrings, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedCBind()  {
		runFrameAppendTest(schemaMixed, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedRBind()  { //note: ncol(A)=ncol(B)
		runFrameAppendTest(schemaMixed, schemaMixed, AppendType.RBIND);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameAppendTest( ValueType[] schema1, ValueType[] schema2, AppendType atype)
	{
		if(shouldSkipTest())
			return;
		
		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema1.length, -10, 10, 0.9, 2373); 
			double[][] B = getRandomMatrix(rows, schema2.length, -10, 10, 0.9, 129); 
			
			//init data frame 1
			FrameBlock frame1 = new FrameBlock(schema1);
			Object[] row1 = new Object[schema1.length];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<schema1.length; j++ )
					A[i][j] = UtilFunctions.objectToDouble(schema1[j], 
							row1[j] = UtilFunctions.doubleToObject(schema1[j], A[i][j]));
				frame1.appendRow(row1);
			}
			
			//init data frame 2
			FrameBlock frame2 = new FrameBlock(schema2);
			Object[] row2 = new Object[schema2.length];
			for( int i=0; i<rows; i++ ) {
				for( int j=0; j<schema2.length; j++ )
					B[i][j] = UtilFunctions.objectToDouble(schema2[j], 
							row2[j] = UtilFunctions.doubleToObject(schema2[j], B[i][j]));
				frame2.appendRow(row2);
			}
			
			
			//core append operations matrix blocks
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			MatrixBlock mbB = DataConverter.convertToMatrixBlock(B);
			MatrixBlock mbC = mbA.append(mbB, new MatrixBlock(), atype==AppendType.CBIND);
			
			//core append operations frame blocks
			FrameBlock frame3 = frame1.append(frame2, new FrameBlock(), atype==AppendType.CBIND);
			
			//check basic meta data
			if( frame3.getNumRows() != mbC.getNumRows() )
				fail("Wrong number of rows: "+frame3.getNumRows()+", expected: "+mbC.getNumRows());
		
			//check correct values
			ValueType[] lschema = frame3.getSchema();
			for( int i=0; i<rows; i++ ) 
				for( int j=0; j<lschema.length; j++ )	{
					double tmp = UtilFunctions.objectToDouble(lschema[j], frame3.get(i, j));
					if( tmp != mbC.quickGetValue(i, j) )
						fail("Wrong get value for cell ("+i+","+j+"): "+tmp+", expected: "+mbC.quickGetValue(i, j));
				}		
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}

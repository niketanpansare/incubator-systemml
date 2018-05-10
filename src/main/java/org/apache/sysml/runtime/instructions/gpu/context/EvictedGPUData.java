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
package org.apache.sysml.runtime.instructions.gpu.context;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;

/**
 * Wrapper class to store the dense and sparse data in a given format
 */
public class EvictedGPUData {
	public float[] floatVal;
	public double[] doubleVal;
	public int[] rowPtr;
	public int[] colInd;
	
	public EvictedGPUData() {
		
	}
	
	public void save(String filePath) throws IOException {
		if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.FLOAT) {
			try(DataOutputStream os = new DataOutputStream(new FileOutputStream(filePath))) {
				os.writeChar(rowPtr == null ? 'd' : 's');
				os.writeInt(floatVal.length);
				for(float val : floatVal) {
					os.writeFloat(val);
				}
				if(rowPtr != null) {
					for(int val : colInd) {
						os.writeInt(val);
					}
					os.writeInt(rowPtr.length);
					for(int val : rowPtr) {
						os.writeInt(val);
					}
				}
			}
		}
		else if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.DOUBLE) {
			try(DataOutputStream os = new DataOutputStream(new FileOutputStream(filePath))) {
				os.writeChar(rowPtr == null ? 'd' : 's');
				os.writeInt(doubleVal.length);
				for(double val : doubleVal) {
					os.writeDouble(val);
				}
				if(rowPtr != null) {
					for(int val : colInd) {
						os.writeInt(val);
					}
					os.writeInt(rowPtr.length);
					for(int val : rowPtr) {
						os.writeInt(val);
					}
				}
			}
		}
		else {
			throw new IllegalArgumentException("ERROR: Unsupported datatype");
		}
	}
	
	
	public static EvictedGPUData load(String filePath) throws FileNotFoundException, IOException {
		EvictedGPUData ret = new EvictedGPUData();
		try(DataInputStream is = new DataInputStream(new FileInputStream(filePath))) {
			char type = is.readChar();
			int numElems = is.readInt();
			if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.FLOAT) {
				ret.floatVal = new float[numElems];
				for(int i = 0; i < numElems; i++) {
					ret.floatVal[i] = is.readFloat();
				}
			}
			else if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.DOUBLE) {
				ret.doubleVal = new double[numElems];
				for(int i = 0; i < numElems; i++) {
					ret.doubleVal[i] = is.readDouble();
				}
			}
			else {
				throw new IllegalArgumentException("ERROR: Unsupported datatype");
			}
			if(type == 's') {
				ret.colInd = new int[numElems];
				for(int i = 0; i < numElems; i++) {
					ret.colInd[i] = is.readInt();
				}
				ret.rowPtr = new int[is.readInt()];
				for(int i = 0; i < ret.rowPtr.length; i++) {
					ret.rowPtr[i] = is.readInt();
				}
			}
			else if(type != 'd') {
				throw new IOException("Incorrect format: " + filePath);
			}
		}
		new File(filePath).delete(); // Delete the file after loading
		return ret;
	}
	
	public long getSizeInBytes() {
		long ret = (floatVal != null) ? floatVal.length*jcuda.Sizeof.FLOAT : 0;
		ret += (doubleVal != null) ? doubleVal.length*jcuda.Sizeof.DOUBLE : 0;
		ret += (rowPtr != null) ? rowPtr.length*jcuda.Sizeof.INT : 0;
		ret += (colInd != null) ? rowPtr.length*jcuda.Sizeof.INT : 0;
		return ret;
	}
}

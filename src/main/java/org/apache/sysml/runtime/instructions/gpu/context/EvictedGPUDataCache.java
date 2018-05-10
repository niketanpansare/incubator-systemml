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

import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Wrapper class to store the dense and sparse data in a given format
 */
class EvictedGPUData {
	public float[] floatVal;
	public double[] doubleVal;
	public int[] rowPtr;
	public int[] colInd;
}

/**
 * An access ordered LRU Cache Map which conforms to the {@link Map} interface
 * while also providing the ability to get the least recently used entry.
 * 
 * This cache is backed by the file system.
 */
public class EvictedGPUDataCache extends LinkedHashMap<GPUObject, EvictedGPUData> {

	private static final long serialVersionUID = 7078404374799241418L;
	
	private final String filePrefix = "evicted_";
	private final String fileSuffix = ".bin";
	protected long _maxSize;
	protected long _currentSize;
	
	// For faster lookup:
	// Rather new File(...).exists() than _gpuObjects.contains(key) is much faster
	protected HashSet<GPUObject> _gpuObjects = new HashSet<>();

	/**
	 * Creates an access-ordered {@link EvictedGPUDataCache}
	 */
	public EvictedGPUDataCache(long maxSize) {
		// An access-ordered LinkedHashMap is instantiated with the default
		// initial capacity and load factors
		super(16, 0.75f, true);
		this._maxSize = maxSize;
		this._currentSize = 0;
	}
	
	/**
	 * Sabe the evicted data to the specified file
	 * 
	 * @param filePath the path where evicted data should be stored
	 * @param data evicted data
	 * @throws IOException if error occurs
	 */
	private void save(String filePath, EvictedGPUData data) throws IOException {
		if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.FLOAT) {
			try(DataOutputStream os = new DataOutputStream(new FileOutputStream(filePath))) {
				os.writeChar(data.rowPtr == null ? 'd' : 's');
				os.writeInt(data.floatVal.length);
				for(float val : data.floatVal) {
					os.writeFloat(val);
				}
				if(data.rowPtr != null) {
					for(int val : data.colInd) {
						os.writeInt(val);
					}
					os.writeInt(data.rowPtr.length);
					for(int val : data.rowPtr) {
						os.writeInt(val);
					}
				}
			}
		}
		else if(LibMatrixCUDA.sizeOfDataType == jcuda.Sizeof.DOUBLE) {
			try(DataOutputStream os = new DataOutputStream(new FileOutputStream(filePath))) {
				os.writeChar(data.rowPtr == null ? 'd' : 's');
				os.writeInt(data.doubleVal.length);
				for(double val : data.doubleVal) {
					os.writeDouble(val);
				}
				if(data.rowPtr != null) {
					for(int val : data.colInd) {
						os.writeInt(val);
					}
					os.writeInt(data.rowPtr.length);
					for(int val : data.rowPtr) {
						os.writeInt(val);
					}
				}
			}
		}
		else {
			throw new IllegalArgumentException("ERROR: Unsupported datatype");
		}
	}
	
	/**
	 * Load the evicted data from the give file path
	 * 
	 * @param filePath path from where evicted data has to be loaded
	 * @return evicted data
	 * @throws FileNotFoundException if file not found
	 * @throws IOException if error occurs
	 */
	private EvictedGPUData load(String filePath) throws FileNotFoundException, IOException {
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

	@Override
	protected boolean removeEldestEntry(Map.Entry<GPUObject, EvictedGPUData> eldest) {
		if(_currentSize > _maxSize) {
			try {
				save(filePrefix + eldest.getKey().hashCode() + fileSuffix, eldest.getValue());
				_currentSize -= getSizeInBytes(eldest.getValue());
			} catch (IOException e) {
				throw new RuntimeException("Unable to persist the evicted entry to the file system.", e);
			}
			return true;
		}
		return false;
	}

	@Override
	public EvictedGPUData put(GPUObject k, EvictedGPUData v) {
		if(k == null)
			throw new IllegalArgumentException("ERROR: an entry with a null key was tried to be inserted in to the EvictedGPUObjectCache");
		else if(containsKey(k) && get(k) != v)
			throw new IllegalArgumentException("ERROR: Multiple Vs associated with the same key");
		else if(containsKey(k))
			return get(k);
		EvictedGPUData ret = super.put(k, v);
		_currentSize += (v.floatVal != null) ? v.floatVal.length*jcuda.Sizeof.FLOAT : 0;
		_currentSize += (v.doubleVal != null) ? v.doubleVal.length*jcuda.Sizeof.DOUBLE : 0;
		_currentSize += (v.rowPtr != null) ? v.rowPtr.length*jcuda.Sizeof.INT : 0;
		_currentSize += (v.colInd != null) ? v.rowPtr.length*jcuda.Sizeof.INT : 0;
		return ret;
	}
	
	public EvictedGPUData get(GPUObject key) {
		EvictedGPUData v = super.get(key);
		if(v == null) {
			String filePath = filePrefix + key.hashCode() + fileSuffix;
			File f = new File(filePath);
			if(f.exists()) {
				try {
					v = load(filePath);
					_currentSize += getSizeInBytes(v);
				} catch (IOException e) {
					throw new RuntimeException("Unable to load the evicted entry from the file system.", e);
				}
			}
		}
        return v;
    }
	
	@Override
	public EvictedGPUData remove(Object key) {
		EvictedGPUData ret = super.remove(key);
		String filePath = filePrefix + key.hashCode() + fileSuffix;
		File f = new File(filePath);
		if(f.exists()) {
			f.delete(); // Delete the file on removal
		}
		_gpuObjects.remove(key);
		return ret;
	}
	
	@Override
	public void clear() {
		super.clear();
		for(GPUObject gpuObj : _gpuObjects) {
			String filePath = filePrefix + gpuObj.hashCode() + fileSuffix;
			File f = new File(filePath);
			if(f.exists()) {
				f.delete(); // Delete the file on removal
			}
		}
		_gpuObjects.clear();
	}
	
	
	public boolean containsKey(GPUObject key) {
		return super.containsKey(key) || _gpuObjects.contains(key);
	}
	
	private long getSizeInBytes(EvictedGPUData v) {
		long ret = (v.floatVal != null) ? v.floatVal.length*jcuda.Sizeof.FLOAT : 0;
		ret += (v.doubleVal != null) ? v.doubleVal.length*jcuda.Sizeof.DOUBLE : 0;
		ret += (v.rowPtr != null) ? v.rowPtr.length*jcuda.Sizeof.INT : 0;
		ret += (v.colInd != null) ? v.rowPtr.length*jcuda.Sizeof.INT : 0;
		return ret;
	}

}
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class EvictedGPUDataCache implements Map<GPUObject, EvictedGPUData> {
	private final String _filePrefix = "evicted_";
	private final String _fileSuffix = ".bin";
	protected final long _maxSize;
	
	protected long _currentSize;
	protected final HashSet<GPUObject> _allKeys;
	protected final Comparator<GPUObject> _evictionPolicy;
	protected final HashMap<GPUObject, EvictedGPUData> _inMemoryMap;
	protected final HashMap<GPUObject, Long> _sizeMap;

	public EvictedGPUDataCache(long maxSize, Comparator<GPUObject> evictionPolicy) {
		_maxSize = maxSize;
		_currentSize = 0;
		_allKeys = new HashSet<>();
		_evictionPolicy = evictionPolicy;
		_inMemoryMap = new HashMap<>();
		_sizeMap = new HashMap<>();
	}
	
	private String getFilePath(Object key) {
		return _filePrefix + key.hashCode() + _fileSuffix;
	}

	@Override
	public void clear() {
		_allKeys.removeAll(_inMemoryMap.keySet());
		for(GPUObject key : _allKeys) {
			File f = new File(getFilePath(key));
			if(!f.exists())
				throw new RuntimeException("The evicted file is not present: " + getFilePath(key));
			else
				f.delete(); // Delete the file on removal
		}
		_allKeys.clear();
		_inMemoryMap.clear();
		_currentSize = 0;
		_sizeMap.clear();
	}

	@Override
	public boolean containsKey(Object key) {
		return _allKeys.contains(key);
	}
	
	private void ensureFreeSize(long size) {
		if(_currentSize + size > _maxSize) {
			List<GPUObject> toEvict = new ArrayList<>(_inMemoryMap.keySet());
			Collections.sort(toEvict, _evictionPolicy);
			while(toEvict.size() > 0 && _currentSize < _maxSize) {
				if(_inMemoryMap.size() == 0)
					throw new RuntimeException("Eviction not possible for " + _currentSize + ", max size:" + _maxSize);
				GPUObject key = toEvict.get(toEvict.size()-1);
				EvictedGPUData value = _inMemoryMap.remove(key);
				try {
					value.save(getFilePath(key));
				} catch (IOException e) {
					throw new RuntimeException("Error while evicting to file system", e);
				}
				_currentSize += value.getSizeInBytes();
			}
		}
	}

	@Override
	public EvictedGPUData get(Object key) {
		if(!_inMemoryMap.containsKey(key) && _allKeys.contains(key)) {	
			try {
				ensureFreeSize(_sizeMap.get(key));
				put((GPUObject)key, EvictedGPUData.load(getFilePath(key)));
			} catch (IOException e) {
				throw new RuntimeException("Unable to get the object", e);
			}
		}
		return _inMemoryMap.get(key);
	}

	@Override
	public boolean isEmpty() {
		return _allKeys.isEmpty();
	}

	@Override
	public Set<GPUObject> keySet() {
		return _allKeys;
	}

	@Override
	public EvictedGPUData put(GPUObject key, EvictedGPUData value) {
		ensureFreeSize(_sizeMap.get(key));
		_inMemoryMap.put(key, value);
		_allKeys.add(key);
		_sizeMap.put(key, value.getSizeInBytes());
		return value;
	}
	
	public void clear(Object key) {
		if(_inMemoryMap.containsKey(key)) {
			_currentSize += _inMemoryMap.remove(key).getSizeInBytes(); 
		}
		else {
			new File(getFilePath(key)).delete();
		}
		_sizeMap.remove(key);
		_allKeys.remove(key);
	}
	

	@Override
	public void putAll(Map<? extends GPUObject, ? extends EvictedGPUData> m) {
		long size = 0;
		for(EvictedGPUData v : m.values()) {
			size += v.getSizeInBytes();
		}
		ensureFreeSize(size);
		for(java.util.Map.Entry<? extends GPUObject, ? extends EvictedGPUData> entry : m.entrySet()) {
			put(entry.getKey(), entry.getValue());
		}
	}

	@Override
	public EvictedGPUData remove(Object key) {
		throw new RuntimeException("Unsupported method"); // requires loading
	}
	
	@Override
	public int size() {
		return _allKeys.size();
	}

	@Override
	public Collection<EvictedGPUData> values() {
		throw new RuntimeException("Unsupported method"); // requires loading
	}
	@Override
	public boolean containsValue(Object value) {
		throw new RuntimeException("Unsupported method"); // requires loading
	}

	@Override
	public Set<java.util.Map.Entry<GPUObject, EvictedGPUData>> entrySet() {
		throw new RuntimeException("Unsupported method"); // requires loading
	}

}
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

import java.util.Comparator;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import jcuda.Pointer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;

public class GPUMatrixMemoryManager {
	protected static final Log LOG = LogFactory.getLog(GPUMatrixMemoryManager.class.getName());
	GPUMemoryManager gpuManager;
	public GPUMatrixMemoryManager(GPUMemoryManager gpuManager) {
		this.gpuManager = gpuManager;
	}
	
	/**
	 * Adds the GPU object to the memory manager
	 * 
	 * @param gpuObj the handle to the GPU object
	 */
	void addGPUObject(GPUObject gpuObj) {
		gpuObjects.add(gpuObj);
	}
	
	/**
	 * Returns worst-case contiguous memory size
	 * @param gpuObj gpu object
	 * @return memory size in bytes
	 */
	private long getWorstCaseContiguousMemorySize(GPUObject gpuObj) {
		long ret = 0;
		if(gpuObj.getJcudaDenseMatrixPtr() != null) {
			ret = gpuManager.allPointers.get(gpuObj.getJcudaDenseMatrixPtr()).getSizeInBytes();
		}
		else if(gpuObj.getJcudaSparseMatrixPtr() != null) {
			CSRPointer sparsePtr = gpuObj.getJcudaSparseMatrixPtr();
			if(sparsePtr.rowPtr != null)
				ret = Math.max(ret, gpuManager.allPointers.get(sparsePtr.rowPtr).getSizeInBytes());
			if(sparsePtr.colInd != null)
				ret = Math.max(ret, gpuManager.allPointers.get(sparsePtr.colInd).getSizeInBytes());
			if(sparsePtr.val != null)
				ret = Math.max(ret, gpuManager.allPointers.get(sparsePtr.val).getSizeInBytes());
		}
		return ret;
	}
	
	/**
	 * Get list of all Pointers in a GPUObject 
	 * @param gObj gpu object 
	 * @return set of pointers
	 */
	Set<Pointer> getPointers(GPUObject gObj) {
		Set<Pointer> ret = new HashSet<>();
		if(gObj.getJcudaDenseMatrixPtr() != null && gObj.getSparseMatrixCudaPointer() != null) {
			LOG.warn("Matrix allocated in both dense and sparse format");
		}
		if(gObj.getJcudaDenseMatrixPtr() != null) {
			ret.add(gObj.getJcudaDenseMatrixPtr());
		}
		if(gObj.getSparseMatrixCudaPointer() != null) {
			CSRPointer sparsePtr = gObj.getSparseMatrixCudaPointer();
			if(sparsePtr != null) {
				if(sparsePtr.rowPtr != null)
					ret.add(sparsePtr.rowPtr);
				else if(sparsePtr.colInd != null)
					ret.add(sparsePtr.colInd);
				else if(sparsePtr.val != null)
					ret.add(sparsePtr.val);
			}
		}
		return ret;
	}
	
	/**
	 * list of allocated {@link GPUObject} instances allocated on {@link GPUContext#deviceNum} GPU
	 * These are matrices allocated on the GPU on which rmvar hasn't been called yet.
	 * If a {@link GPUObject} has more than one lock on it, it cannot be freed
	 * If it has zero locks on it, it can be freed, but it is preferrable to keep it around
	 * so that an extraneous host to dev transfer can be avoided
	 */
	HashSet<GPUObject> gpuObjects = new HashSet<>();
	
	/**
	 * Get GPUObjects from the first memory sections "Matrix Memory"
	 * @param locked return locked GPU objects if true
	 * @param dirty return dirty GPU objects if true
	 * @return set of GPU Objects
	 */
	Set<GPUObject> getGPUObjects(boolean locked, boolean dirty) {
		return gpuObjects.stream().filter(gObj -> gObj.isLocked() == locked && gObj.isDirty() == dirty).collect(Collectors.toSet());
	}
	
	/**
	 * Return all pointers in the first section
	 * @return all pointers in this section
	 */
	Set<Pointer> getPointers() {
		return gpuObjects.stream().flatMap(gObj -> getPointers(gObj).stream()).collect(Collectors.toSet());
	}
	
	/**
	 * Get pointers from the first memory sections "Matrix Memory"
	 * @param locked return locked pointers if true
	 * @param dirty return dirty pointers if true
	 * @return set of pointers
	 */
	Set<Pointer> getPointers(boolean locked, boolean dirty) {
		return gpuObjects.stream().filter(gObj -> gObj.isLocked() == locked && gObj.isDirty() == dirty).flatMap(gObj -> getPointers(gObj).stream()).collect(Collectors.toSet());
	}
	
	/**
	 * Clear the memory of the gpu object that matches the provided parameters
	 * 
	 * @param locked is locked
	 * @param dirty is dirty
	 * @param minSize of atleast given size
	 * @param comparator sorting comparator in case there are more than one gpu object that matches above parameters
	 * @param opcode instruction code
	 * @return true if a gpu object satisfies the above condition else false
	 * @throws DMLRuntimeException if error occurs
	 */
	boolean clear(boolean locked, boolean dirty, long minSize, Comparator<GPUObject> comparator, String opcode) throws DMLRuntimeException {
		Optional<GPUObject> toClear = getGPUObjects(locked, dirty).stream()
				.filter(gObj -> getWorstCaseContiguousMemorySize(gObj) >= minSize)
					.max(comparator);
		if(toClear.isPresent()) {
			GPUObject gObj = toClear.get();
			if(gObj.dirty) 
				gObj.copyFromDeviceToHost(opcode, true, true); // Perform eviction if dirty
			else
				gObj.clearData(opcode, true);
			gpuObjects.remove(gObj);
		}
		return toClear.isPresent();
	}
	
	/**
	 * Clear all unlocked gpu objects
	 * 
	 * @param opcode instruction code
	 * @throws DMLRuntimeException if error
	 */
	void clearAllUnlocked(String opcode) throws DMLRuntimeException {
		Set<GPUObject> unlockedGPUObjects = gpuObjects.stream()
				.filter(gpuObj -> !gpuObj.isLocked()).collect(Collectors.toSet());
		if(unlockedGPUObjects.size() > 0) {
			if(LOG.isWarnEnabled())
				LOG.warn("Clearing all unlocked matrices (count=" + unlockedGPUObjects.size() + ").");
			for(GPUObject toBeRemoved : unlockedGPUObjects) {
				if(toBeRemoved.dirty)
					toBeRemoved.copyFromDeviceToHost(opcode, true, true);
				else
					toBeRemoved.clearData(opcode, true);
			}
			gpuObjects.removeAll(unlockedGPUObjects);
		}
	}
}
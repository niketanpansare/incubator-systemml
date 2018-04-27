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

import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.Collectors;
import java.util.Optional;
import java.util.Set;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;

import jcuda.Pointer;

public class GPULazyCudaFreeMemoryManager {
	GPUMemoryManager gpuManager;
	public GPULazyCudaFreeMemoryManager(GPUMemoryManager gpuManager) {
		this.gpuManager = gpuManager;
	}

	/**
	 * Map of free blocks allocate on GPU. maps size_of_block -> pointer on GPU
	 */
	private HashMap<Long, Set<Pointer>> rmvarGPUPointers = new HashMap<Long, Set<Pointer>>();
	
	/**
	 * Get any pointer of the given size from rmvar-ed pointers (applicable if eager cudaFree is set to false)
	 * 
	 * @param opcode opcode
	 * @param size size in bytes
	 * @return pointer
	 */
	public Pointer getRmvarPointer(String opcode, long size) {
		if (rmvarGPUPointers.containsKey(size)) {
			if(GPUMemoryManager.LOG.isTraceEnabled())
				GPUMemoryManager.LOG.trace("Getting rmvar-ed pointers for size:" + size);
			Pointer A = remove(rmvarGPUPointers, size); // remove from rmvarGPUPointers as you are not calling cudaFree
			return A;
		}
		else {
			return null;
		}
	}
	
	public Set<Pointer> getAllPointers() {
		return rmvarGPUPointers.values().stream().flatMap(ptrs -> ptrs.stream()).collect(Collectors.toSet());
	}
	
	public void clearAll() {
		Set<Pointer> toFree = new HashSet<Pointer>();
		for(Set<Pointer> ptrs : rmvarGPUPointers.values()) {
			toFree.addAll(ptrs);
		}
		rmvarGPUPointers.clear();
		for(Pointer ptr : toFree) {
			gpuManager.guardedCudaFree(ptr);
		}
	}
	
	public Pointer getRmvarPointerMinSize(String opcode, long minSize) throws DMLRuntimeException {
		Optional<Long> toClear = rmvarGPUPointers.entrySet().stream().filter(e -> e.getValue().size() > 0).map(e -> e.getKey())
				.filter(size -> size >= minSize).min((s1, s2) -> s1 < s2 ? -1 : 1);
		if(toClear.isPresent()) {
			long t0 = opcode != null && DMLScript.FINEGRAINED_STATISTICS ?  System.nanoTime() : 0;
			Pointer A = remove(rmvarGPUPointers, toClear.get()); // remove from rmvarGPUPointers as you are not calling cudaFree
			gpuManager.addMiscTime(opcode, GPUInstruction.MISC_TIMER_REUSE, t0);
			return A;
		}
		return null;
	}
	
	
	/**
	 * Remove any pointer in the given hashmap
	 * 
	 * @param hm hashmap of size, pointers
	 * @param size size in bytes
	 * @return the pointer that was removed
	 */
	private Pointer remove(HashMap<Long, Set<Pointer>> hm, long size) {
		Pointer A = hm.get(size).iterator().next();
		remove(hm, size, A);
		return A;
	}
	
	/**
	 * Remove a specific pointer in the given hashmap
	 * 
	 * @param hm hashmap of size, pointers
	 * @param size size in bytes
	 * @param ptr pointer to be removed
	 */
	private void remove(HashMap<Long, Set<Pointer>> hm, long size, Pointer ptr) {
		hm.get(size).remove(ptr);
		if (hm.get(size).isEmpty())
			hm.remove(size);
	}
	
	/**
	 * Return the total memory in bytes used by this memory manager
	 * @return number of bytes
	 */
	public long getTotalMemoryAllocated() {
		long rmvarMemoryAllocated = 0;
		for(long numBytes : rmvarGPUPointers.keySet()) {
			rmvarMemoryAllocated += numBytes;
		}
		return rmvarMemoryAllocated;
	}
	
	/**
	 * Get total number of rmvared pointers
	 * 
	 * @return number of pointers
	 */
	public int getNumPointers() {
		return rmvarGPUPointers.size();
	}
	
	/**
	 * Add a pointer to the rmvar-ed list
	 * @param size size of the pointer
	 * @param toFree pointer
	 */
	public void add(long size, Pointer toFree) {
		Set<Pointer> freeList = rmvarGPUPointers.get(size);
		if (freeList == null) {
			freeList = new HashSet<Pointer>();
			rmvarGPUPointers.put(size, freeList);
		}
		if (freeList.contains(toFree))
			throw new RuntimeException("GPU : Internal state corrupted, double free");
		freeList.add(toFree);
	}
	
	/**
	 * Remove a specific pointer if present in the given hashmap
	 * 
	 * @param hm hashmap of size, pointers
	 * @param size size in bytes
	 * @param ptr pointer to be removed
	 */
	public void removeIfPresent(long size, Pointer ptr) {
		if(rmvarGPUPointers.containsKey(size) && rmvarGPUPointers.get(size).contains(ptr)) {
			rmvarGPUPointers.get(size).remove(ptr);
			if (rmvarGPUPointers.get(size).isEmpty())
				rmvarGPUPointers.remove(size);
		}
	}
}

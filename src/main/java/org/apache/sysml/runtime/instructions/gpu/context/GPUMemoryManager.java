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

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemGetInfo;
import static jcuda.runtime.JCuda.cudaMemset;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.matrix.data.LibMatrixCUDA;
import org.apache.sysml.utils.GPUStatistics;

import jcuda.Pointer;

/**
 * - All cudaFree and cudaMalloc in SystemML should go through this class to avoid OOM or incorrect results.
 * - This class can be refactored in future to accept a chunk of memory ahead of time rather than while execution. This will only thow memory-related errors during startup.  
 */
public class GPUMemoryManager {
	protected static final Log LOG = LogFactory.getLog(GPUMemoryManager.class.getName());
	
	/**
	 * Utility to debug memory leaks
	 */
	private static class PointerInfo {
		private long sizeInBytes;
		private StackTraceElement[] stackTraceElements;
		public PointerInfo(long sizeInBytes) {
			if(DMLScript.PRINT_GPU_MEMORY_INFO) {
				this.stackTraceElements = Thread.currentThread().getStackTrace();
			}
			this.sizeInBytes = sizeInBytes;
		}
		public long getSizeInBytes() {
			return sizeInBytes;
		}
	}
	
	// If the available free size is less than this factor, GPUMemoryManager will warn users of multiple programs grabbing onto GPU memory.
	// This often happens if user tries to use both TF and SystemML, and TF grabs onto 90% of the memory ahead of time.
	private static final double WARN_UTILIZATION_FACTOR = 0.7;
	
	// Invoke cudaMemGetInfo to get available memory information. Useful if GPU is shared among multiple application.
	public double GPU_MEMORY_UTILIZATION_FACTOR = ConfigurationManager.getDMLConfig()
			.getDoubleValue(DMLConfig.GPU_MEMORY_UTILIZATION_FACTOR);
	
	/**
	 * Map of free blocks allocate on GPU. maps size_of_block -> pointer on GPU
	 */
	private HashMap<Long, Set<Pointer>> rmvarGPUPointers = new HashMap<Long, Set<Pointer>>();
	
	/**
	 * list of allocated {@link GPUObject} instances allocated on {@link GPUContext#deviceNum} GPU
	 * These are matrices allocated on the GPU on which rmvar hasn't been called yet.
	 * If a {@link GPUObject} has more than one lock on it, it cannot be freed
	 * If it has zero locks on it, it can be freed, but it is preferrable to keep it around
	 * so that an extraneous host to dev transfer can be avoided
	 */
	private LinkedList<GPUObject> allocatedGPUObjects = new LinkedList<>();
	
	/**
	 * To record size of allocated blocks
	 */
	private HashMap<Pointer, PointerInfo> allocatedGPUPointers = new HashMap<>();
	
	/**
	 * Adds the GPU object to the memory manager
	 * 
	 * @param gpuObj the handle to the GPU object
	 */
	public void addGPUObject(GPUObject gpuObj) {
		allocatedGPUObjects.add(gpuObj);
	}
	
	/**
	 * Get size of allocated GPU Pointer
	 * @param ptr pointer to get size of
	 * @return either the size or -1 if no such pointer exists
	 */
	public long getSizeAllocatedGPUPointer(Pointer ptr) {
		if(allocatedGPUPointers.containsKey(ptr)) {
			return allocatedGPUPointers.get(ptr).getSizeInBytes();
		}
		return -1;
	}
	
	public GPUMemoryManager(GPUContext gpuCtx) {
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		if(free[0] < WARN_UTILIZATION_FACTOR*total[0]) {
			LOG.warn("Potential under-utilization: GPU memory - Total: " + (total[0] * (1e-6)) + " MB, Available: " + (free[0] * (1e-6)) + " MB on " + gpuCtx 
					+ ". This can happen if there are other processes running on the GPU at the same time.");
		}
		else {
			LOG.info("GPU memory - Total: " + (total[0] * (1e-6)) + " MB, Available: " + (free[0] * (1e-6)) + " MB on " + gpuCtx);
		}
		if (GPUContextPool.initialGPUMemBudget() > OptimizerUtils.getLocalMemBudget()) {
			LOG.warn("Potential under-utilization: GPU memory (" + GPUContextPool.initialGPUMemBudget()
					+ ") > driver memory budget (" + OptimizerUtils.getLocalMemBudget() + "). "
					+ "Consider increasing the driver memory budget.");
		}
	}
	
	/**
	 * Invoke cudaMalloc
	 * 
	 * @param A pointer
	 * @param size size in bytes
	 * @return allocated pointer
	 */
	private Pointer cudaMallocWarnIfFails(Pointer A, long size) {
		try {
			cudaMalloc(A, size);
			allocatedGPUPointers.put(A, new PointerInfo(size));
			return A;
		} catch(jcuda.CudaException e) {
			LOG.warn("cudaMalloc failed immediately after cudaMemGetInfo reported that memory of size " + size + " is available. "
					+ "This usually happens if there are external programs trying to grab on to memory in parallel.");
			return null;
		}
	}
	
	/**
	 * Invoke cudaMalloc
	 * 
	 * @param A pointer
	 * @param size size in bytes
	 * @return allocated pointer
	 */
	private Pointer cudaMallocNoWarn(Pointer A, long size) {
		try {
			cudaMalloc(A, size);
			allocatedGPUPointers.put(A, new PointerInfo(size));
			return A;
		} catch(jcuda.CudaException e) {
			return null;
		}
	}
	
	private long getWorstCaseContiguousMemorySize(GPUObject gpuObj) {
		if(gpuObj.getJcudaSparseMatrixPtr() != null) {
			if(gpuObj.getJcudaSparseMatrixPtr().rowPtr != null) {
				return Math.max(gpuObj.getJcudaSparseMatrixPtr().nnz*LibMatrixCUDA.sizeOfDataType, getSizeAllocatedGPUPointer(gpuObj.getJcudaSparseMatrixPtr().rowPtr));
			}
			else {
				return 0;
			}	
		}
		else {
			return gpuObj.getSizeOnDevice();
		}
	}
	
	private String getCallerInfo(StackTraceElement [] stackTrace, int index) {
		if(stackTrace.length <= index)
			return "->";
		else
			return "->" + stackTrace[index].getClassName() + "." + stackTrace[index].getMethodName() + "(" + stackTrace[index].getFileName() + ":" + stackTrace[index].getLineNumber() + ")";
	}
	
	private ArrayList<Pointer> getAllocatedPointers(GPUObject gpuObj) {
		ArrayList<jcuda.Pointer> ret = new ArrayList<>();
		if(gpuObj.getJcudaDenseMatrixPtr() != null) {
			ret.add(gpuObj.getJcudaDenseMatrixPtr());
		} else
			try {
				if(gpuObj.isAllocated() && !gpuObj.isSparseAndEmpty()) {
					ret.add(gpuObj.getJcudaSparseMatrixPtr().rowPtr);
					ret.add(gpuObj.getJcudaSparseMatrixPtr().colInd);
					ret.add(gpuObj.getJcudaSparseMatrixPtr().val);
				}
			} catch (DMLRuntimeException e) {
				throw new RuntimeException(e);
			}
		return ret;
	}
	
	/**
	 * Allocate pointer of the given size in bytes.
	 * 
	 * @param opcode instruction name
	 * @param size size in bytes
	 * @return allocated pointer
	 * @throws DMLRuntimeException if error
	 */
	public Pointer malloc(String opcode, long size) throws DMLRuntimeException {
		if(size < 0) {
			throw new DMLRuntimeException("Cannot allocate memory of size " + size);
		}
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Step 1: First try reusing exact match in rmvarGPUPointers to avoid holes in the GPU memory
		Pointer A = getRmvarPointer(opcode, size);
		Pointer tmpA = (A == null) ? new Pointer() : null;
		// Step 2: Allocate a new pointer in the GPU memory (since memory is available)
		if(A == null && size <= getAvailableMemory()) {
			A = cudaMallocWarnIfFails(tmpA, size);
			if(LOG.isTraceEnabled()) {
				if(A == null)
					LOG.trace("Couldnot allocate a new pointer in the GPU memory:" + size);
				else
					LOG.trace("Allocated a new pointer in the GPU memory:" + size);
			}
		}
		
		// Reusing one rmvar-ed pointer (Step 3) is preferred to reusing multiple pointers as the latter may not be contiguously allocated.
		// (Step 4 or using any other policy that doesnot take memory into account).
		
		// Step 3: Try reusing non-exact match entry of rmvarGPUPointers
		if(A == null) { 
			// Find minimum key that is greater than size
			long key = Long.MAX_VALUE;
			for(Long k : rmvarGPUPointers.keySet()) {
				key = k > size ? Math.min(key, k) : key;
			}
			if(key != Long.MAX_VALUE) {
				A = getRmvarPointer(opcode, key);
				// To avoid potential for holes in the GPU memory
				guardedCudaFree(A);
				A = cudaMallocWarnIfFails(tmpA, size);
				if(LOG.isTraceEnabled()) {
					if(A == null)
						LOG.trace("Couldnot reuse non-exact match of rmvarGPUPointers:" + size);
					else
						LOG.trace("Reuses a non-exact match from rmvarGPUPointers:" + size);
				}
			}
		}
		
		// Step 3.b: An optimization missing so as not to over-engineer malloc:
		// Try to find minimal number of contiguously allocated pointer.
		
		// Evictions of matrix blocks are expensive (as they might lead them to be written to disk in case of smaller CPU budget) 
		// than doing cuda free/malloc/memset. So, rmvar-ing every blocks (step 4) is preferred to eviction (step 5).
		
		// Step 4: Eagerly free-up rmvarGPUPointers and check if memory is available on GPU
		if(A == null) {
			Set<Pointer> toFree = new HashSet<Pointer>();
			for(Set<Pointer> ptrs : rmvarGPUPointers.values()) {
				toFree.addAll(ptrs);
			}
			for(Pointer ptr : toFree) {
				guardedCudaFree(ptr);
			}
			if(size <= getAvailableMemory()) {
				A = cudaMallocWarnIfFails(tmpA, size);
				if(LOG.isTraceEnabled()) {
					if(A == null)
						LOG.trace("Couldnot allocate a new pointer in the GPU memory after eager free:" + size);
					else
						LOG.trace("Allocated a new pointer in the GPU memory after eager free:" + size);
				}
			}
		}
		
		addMiscTime(opcode, GPUStatistics.cudaAllocTime, GPUStatistics.cudaAllocCount, GPUInstruction.MISC_TIMER_ALLOCATE, t0);
		
		// Step 5: Try eviction based on the given policy
		if(A == null) {
			t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			
			// Deallocate maximum to clear up more memory for the future evictions.
			Optional<GPUObject> toDelete = allocatedGPUObjects.stream()
					.filter(gpuObj -> !gpuObj.isLocked() && !gpuObj.isDirty() && getWorstCaseContiguousMemorySize(gpuObj) >= size)
					.max((o1, o2) -> o1.getSizeOnDevice() < o2.getSizeOnDevice() ? -1: 1);
			if(toDelete.isPresent()) {
				// Delete toDelete from the GPU memory. No eviction to the host memory required as it is not dirty.
				allocatedGPUObjects.remove(toDelete.get());
				toDelete.get().clearData(true);
			}
			else {
				Optional<GPUObject> toEvict = allocatedGPUObjects.stream()
						.filter(gpuObj -> !gpuObj.isLocked() && getWorstCaseContiguousMemorySize(gpuObj) >= size)
						.max(new GPUComparator(size));
				if(toEvict.isPresent()) {
					GPUObject toBeRemoved = toEvict.get();
					// Perform eviction
					if(toBeRemoved.dirty) {
						toBeRemoved.copyFromDeviceToHost(opcode, true);
					}
					allocatedGPUObjects.remove(toBeRemoved);
					toBeRemoved.clearData(true);
				}
				else {
					// Evict all
					List<GPUObject> unlockedGPUObjects = allocatedGPUObjects.stream()
												.filter(gpuObj -> !gpuObj.isLocked()).collect(Collectors.toList());
					allocatedGPUObjects.removeAll(unlockedGPUObjects);
					for(GPUObject toBeRemoved : unlockedGPUObjects) {
						if(toBeRemoved.dirty) {
							toBeRemoved.copyFromDeviceToHost(opcode, true);
						}
						toBeRemoved.clearData(true);
					}
				}
			}
			addMiscTime(opcode, GPUStatistics.cudaEvictionCount, GPUStatistics.cudaEvictTime, GPUInstruction.MISC_TIMER_EVICT, t0);
			A = cudaMallocWarnIfFails(tmpA, size);
		}
		
		// Step 6: Handle defragmentation
		if(A == null) {
			t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			LOG.warn("Potential fragmentation of GPU memory");
			List<GPUObject> unlockedNonDirtyGPUObjects = allocatedGPUObjects.stream()
														.filter(gpuObj -> !gpuObj.isLocked() && !gpuObj.isDirty() 
														&& getWorstCaseContiguousMemorySize(gpuObj) >= size).collect(Collectors.toList());
			allocatedGPUObjects.removeAll(unlockedNonDirtyGPUObjects);
			for(GPUObject toBeRemoved : unlockedNonDirtyGPUObjects) {
				toBeRemoved.clearData(true);
			}
			A = cudaMallocNoWarn(tmpA, size);
			if(A == null) {
				// Evict all
				List<GPUObject> unlockedGPUObjects = allocatedGPUObjects.stream()
											.filter(gpuObj -> !gpuObj.isLocked()).collect(Collectors.toList());
				allocatedGPUObjects.removeAll(unlockedGPUObjects);
				for(GPUObject toBeRemoved : unlockedGPUObjects) {
					if(toBeRemoved.dirty) {
						toBeRemoved.copyFromDeviceToHost(opcode, true);
					}
					toBeRemoved.clearData(true);
				}
				A = cudaMallocNoWarn(tmpA, size);
			}
			addMiscTime(opcode, GPUStatistics.cudaEvictionCount, GPUStatistics.cudaEvictTime, GPUInstruction.MISC_TIMER_EVICT, t0);
		}
		
		if(A == null) {
			String hint = "";
			if(DMLScript.PRINT_GPU_MEMORY_INFO) {
				Set<Pointer> managedPointers = allocatedGPUObjects.stream().flatMap(gpuObj -> getAllocatedPointers(gpuObj).stream()).collect(Collectors.toSet());
				managedPointers.addAll(rmvarGPUPointers.values().stream().flatMap(ptrs -> ptrs.stream()).collect(Collectors.toSet()));
				Set<Pointer> leakedPointers = nonIn(allocatedGPUPointers.keySet(), managedPointers);
				if(leakedPointers.size() > 0) {
					System.out.println("Leaked GPU Pointers were allocated by:");
					for(Pointer leakedPointer : leakedPointers) {
						PointerInfo ptrInfo = allocatedGPUPointers.get(leakedPointer);
						System.out.println(">>" + 
								// getCallerInfo(ptrInfo.stackTraceElements, 5) + getCallerInfo(ptrInfo.stackTraceElements, 6) + getCallerInfo(ptrInfo.stackTraceElements, 7) +
								getCallerInfo(ptrInfo.stackTraceElements, 8) + getCallerInfo(ptrInfo.stackTraceElements, 9) + getCallerInfo(ptrInfo.stackTraceElements, 10));
					}
				}
				else {
					System.out.println("No leaked GPU Pointers were found.");
				}
			}
			else {
				hint = ". Hint: Please turn on the DEBUG_GPU_MEMORY_LEAKS developer flag to debug this issue.";
			}
			throw new DMLRuntimeException("There is not enough memory on device for this matrix, request (" + size + "). "
					+ toString() + hint);
		}
		
		t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		cudaMemset(A, 0, size);
		addMiscTime(opcode, GPUStatistics.cudaMemSet0Time, GPUStatistics.cudaMemSet0Count, GPUInstruction.MISC_TIMER_SET_ZERO, t0);
		return A;
	}
	
	private static Pointer EMPTY_POINTER = new Pointer();
	
	/**
	 * Note: This method should not be called from an iterator as it removes entries from allocatedGPUPointers and rmvarGPUPointers
	 * 
	 * @param toFree pointer to call cudaFree method on
	 */
	private void guardedCudaFree(Pointer toFree) {
		if (toFree != EMPTY_POINTER) {
			if(allocatedGPUPointers.containsKey(toFree)) {
				long size = allocatedGPUPointers.remove(toFree).getSizeInBytes();
				if(rmvarGPUPointers.containsKey(size) && rmvarGPUPointers.get(size).contains(toFree)) {
					remove(rmvarGPUPointers, size, toFree);
				}
				if(LOG.isDebugEnabled())
					LOG.debug("Free-ing up the pointer: " + toFree);
				cudaFree(toFree);
			}
			else {
				throw new RuntimeException("Attempting to free an unaccounted pointer:" + toFree);
			}
		}
	}
	
	/**
	 * Deallocate the pointer
	 * 
	 * @param opcode instruction name
	 * @param toFree pointer to free
	 * @param eager whether to deallocate eagerly
	 * @throws DMLRuntimeException if error
	 */
	public void free(String opcode, Pointer toFree, boolean eager) throws DMLRuntimeException {
		if (toFree == EMPTY_POINTER) { // trying to free a null pointer
			return;
		}
		if (eager) {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			guardedCudaFree(toFree);
			addMiscTime(opcode, GPUStatistics.cudaDeAllocTime, GPUStatistics.cudaDeAllocCount, GPUInstruction.MISC_TIMER_CUDA_FREE, t0);
		}
		else {
			if (!allocatedGPUPointers.containsKey(toFree))
				throw new RuntimeException("ERROR : Internal state corrupted, cache block size map is not aware of a block it trying to free up");
			long size = allocatedGPUPointers.get(toFree).getSizeInBytes();
			Set<Pointer> freeList = rmvarGPUPointers.get(size);
			if (freeList == null) {
				freeList = new HashSet<Pointer>();
				rmvarGPUPointers.put(size, freeList);
			}
			if (freeList.contains(toFree))
				throw new RuntimeException("GPU : Internal state corrupted, double free");
			freeList.add(toFree);
		}
	}
	
	/**
	 * Removes the GPU object from the memory manager
	 * 
	 * @param gpuObj the handle to the GPU object
	 */
	public void removeGPUObject(GPUObject gpuObj) {
		if(LOG.isDebugEnabled())
			LOG.debug("Removing the GPU object: " + gpuObj);
		allocatedGPUObjects.removeIf(a -> a.equals(gpuObj));
	}

	
	/**
	 * Clear the allocated GPU objects
	 * 
	 * @throws DMLRuntimeException if error
	 */
	public void clearMemory() throws DMLRuntimeException {
		// First deallocate all the GPU objects
		for(GPUObject gpuObj : allocatedGPUObjects) {
			if(gpuObj.isDirty()) {
				LOG.debug("Attempted to free GPU Memory when a block[" + gpuObj + "] is still on GPU memory, copying it back to host.");
				gpuObj.acquireHostRead(null);
			}
			gpuObj.clearData(true);
		}
		allocatedGPUObjects.clear();
		
		// Then clean up remaining allocated GPU pointers 
		Set<Pointer> remainingPtr = new HashSet<>(allocatedGPUPointers.keySet());
		for(Pointer toFree : remainingPtr) {
			guardedCudaFree(toFree); // cleans up allocatedGPUPointers and rmvarGPUPointers as well
		}
	}
	
	/**
	 * Get all pointers withing allocatedGPUObjects such that GPUObject is in dirty state
	 * 
	 * @return set of pointers
	 */
	private HashSet<Pointer> getDirtyPointers() {
		HashSet<Pointer> nonTemporaryPointers = new HashSet<Pointer>();
		for (GPUObject o : allocatedGPUObjects) {
			if(o.isDirty()) {
				if (o.isSparse()) {
					CSRPointer p = o.getSparseMatrixCudaPointer();
					if (p == null)
						throw new RuntimeException("CSRPointer is null in clearTemporaryMemory");
					if (p.rowPtr != null) {
						nonTemporaryPointers.add(p.rowPtr);
					}
					if (p.colInd != null) {
						nonTemporaryPointers.add(p.colInd);
					}
					if (p.val != null) {
						nonTemporaryPointers.add(p.val);
					}

				} else {
					Pointer p = o.getJcudaDenseMatrixPtr();
					if (p == null)
						throw new RuntimeException("Pointer is null in clearTemporaryMemory");
					nonTemporaryPointers.add(p);
				}
			}
		}
		
		return nonTemporaryPointers;
	}
	
	/**
	 * Performs a non-in operation
	 * 
	 * @param superset superset of pointer
	 * @param subset subset of pointer
	 * @return pointers such that: superset - subset
	 */
	private Set<Pointer> nonIn(Set<Pointer> superset, Set<Pointer> subset) {
		Set<Pointer> ret = new HashSet<Pointer>();
		for(Pointer superPtr : superset) {
			if(!subset.contains(superPtr)) {
				ret.add(superPtr);
			}
		}
		return ret;
	}
	
	/**
	 * Clears up the memory used by non-dirty pointers.
	 */
	public void clearTemporaryMemory() {
		// To record the cuda block sizes needed by allocatedGPUObjects, others are cleared up.
		Set<Pointer> temporaryPointers = nonIn(allocatedGPUPointers.keySet(), getDirtyPointers());
		for(Pointer tmpPtr : temporaryPointers) {
			guardedCudaFree(tmpPtr);
		}
	}
	
	/**
	 * Convenient method to add misc timers
	 * 
	 * @param opcode opcode
	 * @param globalGPUTimer member of GPUStatistics
	 * @param globalGPUCounter member of GPUStatistics
	 * @param instructionLevelTimer member of GPUInstruction
	 * @param startTime start time
	 */
	private void addMiscTime(String opcode, LongAdder globalGPUTimer, LongAdder globalGPUCounter, String instructionLevelTimer, long startTime) {
		if(DMLScript.STATISTICS) {
			long totalTime = System.nanoTime() - startTime;
			globalGPUTimer.add(totalTime);
			globalGPUCounter.add(1);
			if (opcode != null && DMLScript.FINEGRAINED_STATISTICS)
				GPUStatistics.maintainCPMiscTimes(opcode, instructionLevelTimer, totalTime);
		}
	}
	
	/**
	 * Convenient method to add misc timers
	 * 
	 * @param opcode opcode
	 * @param instructionLevelTimer member of GPUInstruction
	 * @param startTime start time
	 */
	private void addMiscTime(String opcode, String instructionLevelTimer, long startTime) {
		if (opcode != null && DMLScript.FINEGRAINED_STATISTICS)
			GPUStatistics.maintainCPMiscTimes(opcode, instructionLevelTimer, System.nanoTime() - startTime);
	}
	
	/**
	 * Get any pointer of the given size from rmvar-ed pointers (applicable if eager cudaFree is set to false)
	 * 
	 * @param opcode opcode
	 * @param size size in bytes
	 * @return pointer
	 */
	private Pointer getRmvarPointer(String opcode, long size) {
		if (rmvarGPUPointers.containsKey(size)) {
			if(LOG.isTraceEnabled())
				LOG.trace("Getting rmvar-ed pointers for size:" + size);
			long t0 = opcode != null && DMLScript.FINEGRAINED_STATISTICS ?  System.nanoTime() : 0;
			Pointer A = remove(rmvarGPUPointers, size); // remove from rmvarGPUPointers as you are not calling cudaFree
			addMiscTime(opcode, GPUInstruction.MISC_TIMER_REUSE, t0);
			return A;
		}
		else {
			return null;
		}
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
	 * Print debugging information
	 */
	public String toString() {
		long sizeOfLockedGPUObjects = 0; long numLockedGPUObjects = 0;
		long sizeOfUnlockedGPUObjects = 0; long numUnlockedGPUObjects = 0;
		for(GPUObject gpuObj : allocatedGPUObjects) {
			if(gpuObj.isLocked()) {
				numLockedGPUObjects++;
				sizeOfLockedGPUObjects += gpuObj.getSizeOnDevice();
			}
			else {
				numUnlockedGPUObjects++;
				sizeOfUnlockedGPUObjects += gpuObj.getSizeOnDevice();
			}
		}
		long rmvarMemoryAllocated = 0;
		for(long numBytes : rmvarGPUPointers.keySet()) {
			rmvarMemoryAllocated += numBytes;
		}
		long totalMemoryAllocated = 0;
		for(PointerInfo ptrInfo : allocatedGPUPointers.values()) {
			totalMemoryAllocated += ptrInfo.getSizeInBytes();
		}
		return "Num of GPU objects: [unlocked:" + numUnlockedGPUObjects + ", locked:" + numLockedGPUObjects + "]. "
				+ "Size of GPU objects in bytes: [unlocked:" + sizeOfUnlockedGPUObjects + ", locked:" + sizeOfLockedGPUObjects + "]. "
				+ "Total memory allocated by the current GPU context in bytes:" + totalMemoryAllocated + ", number of allocated pointers:" + allocatedGPUPointers.size() + ". "
				+ "Total memory rmvared by the current GPU context in bytes:" + rmvarMemoryAllocated + ", number of rmvared pointers:" + rmvarGPUPointers.size();
	}
	
	/**
	 * Gets the available memory on GPU that SystemML can use.
	 *
	 * @return the available memory in bytes
	 */
	public long getAvailableMemory() {
		long free[] = { 0 };
		long total[] = { 0 };
		cudaMemGetInfo(free, total);
		return (long) (free[0] * GPU_MEMORY_UTILIZATION_FACTOR);
	}
	
	/**
	 * Class that governs the eviction policy
	 */
	public static class GPUComparator implements Comparator<GPUObject> {
		private long neededSize;
		public GPUComparator(long neededSize) {
			this.neededSize = neededSize;
		}
		@Override
		public int compare(GPUObject p1, GPUObject p2) {
			if (p1.isLocked() && p2.isLocked()) {
				// Both are locked, so don't sort
				return 0;
			} else if (p1.isLocked()) {
				// Put the unlocked one to RHS
				// a value less than 0 if x < y; and a value greater than 0 if x > y
				return -1;
			} else if (p2.isLocked()) {
				// Put the unlocked one to RHS
				// a value less than 0 if x < y; and a value greater than 0 if x > y
				return 1;
			} else {
				// Both are unlocked
				if (DMLScript.GPU_EVICTION_POLICY == DMLScript.EvictionPolicy.MIN_EVICT) {
					long p1Size = p1.getSizeOnDevice() - neededSize;
					long p2Size = p2.getSizeOnDevice() - neededSize;

					if (p1Size >= 0 && p2Size >= 0) {
						return Long.compare(p2Size, p1Size);
					} else {
						return Long.compare(p1Size, p2Size);
					}
				} else {
					return Long.compare(p2.timestamp.get(), p1.timestamp.get());
				}
			}
		}
	}
}

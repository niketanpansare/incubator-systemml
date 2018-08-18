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

package org.apache.sysml.api;

import java.util.HashSet;
import java.util.List;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.codegen.SpoofCompiler;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysml.utils.Statistics;

public class ScriptExecutorUtils {
	
	public static enum SystemMLAPI {
		DMLScript,
		MLContext,
		JMLC
	}
	
	private static List<GPUContext> gCtxs = null;
	public static List<GPUContext> reserveAllGPUContexts() {
		gCtxs = GPUContextPool.reserveAllGPUContexts();
		if (gCtxs == null) {
			throw new DMLRuntimeException(
					"Could not create GPUContext, either no GPU or all GPUs currently in use");
		}
		return gCtxs;
	}
	
	public static void freeAllGPUContexts() {
		if(gCtxs == null) {
			throw new DMLRuntimeException("Trying to free GPUs without reserving them first.");
		}
		for(GPUContext gCtx : gCtxs) {
			gCtx.clearTemporaryMemory();
		}
		GPUContextPool.freeAllGPUContexts();
		gCtxs = null;
	}

	/**
	 * Execute the runtime program. This involves execution of the program
	 * blocks that make up the runtime program and may involve dynamic
	 * recompilation.
	 * 
	 * @param rtprog
	 *            runtime program
	 * @param dmlconf
	 *            dml configuration
	 * @param statisticsMaxHeavyHitters
	 *            maximum number of statistics to print
	 * @param symbolTable
	 *            symbol table (that were registered as input as part of MLContext)
	 * @param outputVariables
	 *            output variables (that were registered as output as part of MLContext)
	 * @param api
	 * 			  API used to execute the runtime program
	 * @param gCtxs
	 * 			  list of GPU contexts
	 * @return execution context
	 */
	public static ExecutionContext executeRuntimeProgram(Program rtprog, DMLConfig dmlconf, int statisticsMaxHeavyHitters, 
			LocalVariableMap symbolTable, HashSet<String> outputVariables, SystemMLAPI api, List<GPUContext> gCtxs) {
		boolean exceptionThrown = false;
		// Start timer (disabled for JMLC)
		if(api != SystemMLAPI.JMLC)
			Statistics.startRunTimer();
		
		// Create execution context and attach registered outputs
		ExecutionContext ec = ExecutionContextFactory.createContext(symbolTable, rtprog);
		if(outputVariables != null)
			ec.getVariables().setRegisteredOutputs(outputVariables);
		
		// Assign GPUContext to the current ExecutionContext
		if(gCtxs != null) {
			gCtxs.get(0).initializeThread();
			ec.setGPUContexts(gCtxs);
		}
		
		try {
			// run execute (w/ exception handling to ensure proper shutdown)
			rtprog.execute(ec);
		} catch (Throwable e) {
			exceptionThrown = true;
			throw e;
		} finally { // ensure cleanup/shutdown
			if (DMLScript.USE_ACCELERATOR && !ec.getGPUContexts().isEmpty()) {
				// -----------------------------------------------------------------
				// The below code pulls the output variables on the GPU to the host. This is required especially when:
				// The output variable was generated as part of a MLContext session with GPU enabled
				// and was passed to another MLContext with GPU disabled
				// The above scenario occurs in our gpu test suite (eg: BatchNormTest).
				if(outputVariables != null) {
					for(String outVar : outputVariables) {
						Data data = ec.getVariable(outVar);
						if(data != null && data instanceof MatrixObject) {
							for(GPUContext gCtx : ec.getGPUContexts()) {
								GPUObject gpuObj = ((MatrixObject)data).getGPUObject(gCtx);
								if(gpuObj != null && gpuObj.isDirty()) {
									gpuObj.acquireHostRead(null);
								}
							}
						}
					}
				}
				// -----------------------------------------------------------------
			}
			if( ConfigurationManager.isCodegenEnabled() )
				SpoofCompiler.cleanupCodeGenerator();
			
			//cleanup unnecessary outputs
			symbolTable.removeAllNotIn(outputVariables);
			
			// Display statistics (disabled for JMLC)
			if(api != SystemMLAPI.JMLC) {
				Statistics.stopRunTimer();
				(exceptionThrown ? System.err : System.out)
					.println(Statistics.display(statisticsMaxHeavyHitters > 0 ?
						statisticsMaxHeavyHitters : DMLScript.STATISTICS_COUNT));
			}
		}
		return ec;
	}

}

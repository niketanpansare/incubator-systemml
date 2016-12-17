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

package org.apache.sysml.runtime.controlprogram;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;


/**
 * 
 * 1. Compile the jar: 
 * mvn package
 * 
 * 2. Compile headers (only for development):
 * javah -classpath SystemML.jar org.apache.sysml.runtime.controlprogram.CPPUtil
 * 
 * 3. 
 */
public class CPPUtil {
	
	private static final Log LOG = LogFactory.getLog(CPPUtil.class.getName());
	private static boolean libraryLoaded = false;
	static {
		// Load native library at runtime
		// SystemML.dll (Windows) or libSystemML.so (Unix)
		if(DMLScript.ENABLE_NATIVE_BLAS) {
			try {
				System.loadLibrary("systemml");
				libraryLoaded = true;
				LOG.debug("Successfully loaded native library");
			}
			catch (UnsatisfiedLinkError e) { }
		}
	}
	public static boolean isLibraryLoaded() {
		return libraryLoaded;
	}
	
	
	public static native void matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen);
	public static native void conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q);
}

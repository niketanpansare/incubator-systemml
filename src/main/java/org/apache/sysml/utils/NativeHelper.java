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

package org.apache.sysml.utils;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.HashMap;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.File;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.SystemUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;

/**
 * This class helps in loading native library.
 * By default, it first tries to load Intel MKL, else tries to load OpenBLAS. 
 * This behavior can be controlled by setting the environment variable SYSTEMML_BLAS to either mkl or openblas.
 */
public class NativeHelper {
	private static boolean isSystemMLLoaded = false;
	private static final Log LOG = LogFactory.getLog(NativeHelper.class.getName());
	private static HashMap<String, String> archMap = new HashMap<String, String>();
	public static String blasType;
	static {
        archMap.put("x86", "x86_32");
        archMap.put("i386", "x86_32");
        archMap.put("i486", "x86_32");
        archMap.put("i586", "x86_32");
        archMap.put("i686", "x86_32");
        archMap.put("x86_64", "x86_64");
        archMap.put("amd64", "x86_64");
        archMap.put("powerpc", "ppc_64");
	
		blasType = getBLASType();
		if(blasType != null) {
			try {
				loadLibrary("systemml", "_" + blasType);
				isSystemMLLoaded = true;
				LOG.info("Using native blas: " + blasType);
			} catch (IOException e) {
				LOG.warn("Using Java-based BLAS as unable to load native BLAS");
			}
		}
	}
	
	private static int maxNumThreads = -1;
	private static boolean setMaxNumThreads = false;
	public static boolean isNativeLibraryLoaded(int numThreads) {
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		if(isSystemMLLoaded && !setMaxNumThreads) {
			setMaxNumThreads(maxNumThreads);
			setMaxNumThreads = true;
		}
		if(DMLScript.ENABLE_NATIVE_BLAS_IN_MULTITHREADED_SCENARIO) {
			return isSystemMLLoaded;
		}
		else {
			return isSystemMLLoaded && (numThreads == maxNumThreads);
		}
	}
	
	public static int getMaxNumThreads() {
		if(maxNumThreads == -1)
			maxNumThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		return maxNumThreads;
	}
	
	private static String getBLASType() {
		String specifiedBLAS = System.getenv("SYSTEMML_BLAS");
		if(specifiedBLAS != null) {
			if(specifiedBLAS.trim().toLowerCase().equals("mkl")) {
				return isMKLAvailable() ? "mkl" : null;
			}
			else if(specifiedBLAS.trim().toLowerCase().equals("openblas")) {
				return isOpenBLASAvailable() ? "openblas" : null;
			}
			else if(specifiedBLAS.trim().toLowerCase().equals("none")) {
				LOG.warn("Not loading native BLAS as SYSTEMML_BLAS=" + specifiedBLAS);
				return null;
			}
			else {
				LOG.warn("Unknown BLAS:" + specifiedBLAS);
				return null;
			}
		}
		else {
			// No BLAS specified ... try loading Intel MKL first
			return isMKLAvailable() ? "mkl" : (isOpenBLASAvailable() ? "openblas" : null);
		}
	}
	
	private static boolean isPreloadSystemML = false;
	private static boolean isMKLAvailable() {
		try {
			// ------------------------------------------------------------
			// Set environment variable MKL_THREADING_LAYER to GNU on Linux for performance
			if(SystemUtils.IS_OS_LINUX) {
				if(!isPreloadSystemML) {
					loadLibrary("preload_systemml", "");
					EnvironmentHelper.setEnv("MKL_THREADING_LAYER", "GNU");
					isPreloadSystemML = true;
				}
				
				try {
					System.loadLibrary("gomp");
				}
				catch (UnsatisfiedLinkError e) {
					LOG.warn("Unable to load mkl: GNU OpenMP (libgomp) required for loading MKL-enabled  SystemML library" + e.getMessage());
					return false;
				}
			}
			// ------------------------------------------------------------
			System.loadLibrary("mkl_rt");
			return true;
		}
		catch (UnsatisfiedLinkError e) {
			LOG.warn("Unable to load mkl:" + e.getMessage());
			return false;
		} catch (IOException e) {
			LOG.warn("Unable to load preload_systemml required for mkl:" + e.getMessage());
			return false;
		}
	}
	
	private static boolean isOpenBLASAvailable() {
		if(SystemUtils.IS_OS_WINDOWS) {
			String message = "";
			try {
				 System.loadLibrary("openblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				message += e.getMessage() + " ";
			}
			try {
				 System.loadLibrary("libopenblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				message += e.getMessage() + " ";
			}
			LOG.warn("Unable to load openblas:" + message);
			return false;
		}
		else {
			try {
				 System.loadLibrary("openblas");
				 return true;
			}
			catch (UnsatisfiedLinkError e) {
				LOG.warn("Unable to load openblas:" + e.getMessage());
				return false;
			}
		}
	}
	
	private static void loadLibrary(String libName, String suffix1) throws IOException {
		String prefix = "";
		String suffix2 = "";
		String os = "";
		if (SystemUtils.IS_OS_MAC_OSX) {
			prefix = "lib";
			suffix2 = "dylib";
			os = "apple";
		} else if (SystemUtils.IS_OS_LINUX) {
			prefix = "lib";
			suffix2 = "so";
			os = "linux";
		} else if (SystemUtils.IS_OS_WINDOWS) {
			prefix = "";
			suffix2 = "dll";
			os = "windows";
		} else {
			LOG.info("Unsupported OS:" + SystemUtils.OS_NAME);
			throw new IOException("Unsupported OS");
		}
		
		String arch = archMap.get(SystemUtils.OS_ARCH);
		if(arch == null) {
			LOG.info("Unsupported architecture:" + SystemUtils.OS_ARCH);
			throw new IOException("Unsupported architecture:" + SystemUtils.OS_ARCH);
		}
		loadLibraryHelper(prefix + libName + suffix1 + "-" + os + "-" + arch + "." + suffix2);
	}

	private static void loadLibraryHelper(String path) throws IOException {
		InputStream in = null; OutputStream out = null;
		try {
			in = NativeHelper.class.getResourceAsStream("/lib/"+path);
			if(in != null) {
				File temp = File.createTempFile(path, "");
				temp.deleteOnExit();
				out = FileUtils.openOutputStream(temp);
		        IOUtils.copy(in, out);
		        in.close(); in = null;
		        out.close(); out = null;
				System.load(temp.getAbsolutePath());
			}
			else
				throw new IOException("No lib available in the jar:" + path);
			
		} catch(IOException e) {
			LOG.info("Unable to load library " + path + " from resource:" + e.getMessage());
			throw e;
		} finally {
			if(out != null)
				out.close();
			if(in != null)
				in.close();
		}
		
	}
	
	// TODO: Add pmm, wsloss, mmchain, etc.
	public static native boolean matrixMultDenseDense(double [] m1, double [] m2, double [] ret, int m1rlen, int m1clen, int m2clen, int numThreads);
	public static native boolean tsmm(double [] m1, double [] ret, int m1rlen, int m1clen, boolean isLeftTranspose, int numThreads);
	
	// LibMatrixDNN operations:
	// N = number of images, C = number of channels, H = image height, W = image width
	// K = number of filters, R = filter height, S = filter width
	public static native boolean conv2dDense(double [] input, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBiasAddDense(double [] input, double [] bias, double [] filter, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBackwardDataDense(double [] filter, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	public static native boolean conv2dBackwardFilterDense(double [] input, double [] dout, double [] ret, int N, int C, int H, int W, 
			int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q, int numThreads);
	
	private static native void setMaxNumThreads(int numThreads);
}

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
 
#include "libmatrixmult.h"
#include <cstdlib>
#include "omp.h"
#include <cmath>

int SYSML_CURRENT_NUM_THREADS = -1;
// This method is called during every systemml call, but the if condition is only executed once.
void ensureSequentialBLAS() {
#ifdef USE_INTEL_MKL
	if(SYSML_CURRENT_NUM_THREADS != 1) {
	    SYSML_CURRENT_NUM_THREADS = 1;
	    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
	    mkl_set_num_threads(1);
	}
#endif
#ifdef USE_OPEN_BLAS
	if(SYSML_CURRENT_NUM_THREADS != 1) {
		SYSML_CURRENT_NUM_THREADS = 1;
		openblas_set_num_threads(1);
	}
#endif
}

 
// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads) {
  // First step: Avoids oversubscription and other openmp/internal blas threading issues
  ensureSequentialBLAS();
  
  int m = m1rlen;
  int n = m2clen;
  int k = m1clen;
  if(numThreads == 1) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k,
              m2Ptr, n, 0.0, retPtr, n);
  }
  else {
  	// We only use OpenMP for parallelism to avoid any BLAS threading library conflict.
  	// For example: if openblas is not compiled with USE_OPENMP=1 or if Intel MKL is using Intel threading library
  	// In certain cases (Linux + Intel MKL with GNU multithreading layer), this can be avoided 
  	// but for simplifying performance testing we are sticking to this approach.  
  	// See https://software.intel.com/en-us/node/528707
  	
  	// Performance TODO: Use Column-wise parallelism (only if RHS is extremely wide) ... has additional overhead of in-place transpose 
  	// For example: if(m1rlen < numThreads && m2clen > 2*numThreads)
  	//  m1 %*% m2 = t( t(m2) %*% t(m1) )
  	
  	// Row-wise parallelism	
#pragma omp parallel for num_threads(MIN(m1rlen, numThreads))
	for(int i = 0; i < m1rlen; i++) 
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n, k, 1.0, m1Ptr + i*m1clen, k,
          m2Ptr, n, 0.0, retPtr + i*m2clen, n);
	
  	// ------------------------------------------------------------
  }
}
 
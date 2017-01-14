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
 
#ifndef _libmatrixmult_h
#define _libmatrixmult_h

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// *****************************************************************
// We support Intel MKL (recommended) or OpenBLAS.
// These flags are used for conditional compilation with mkl and openblas
#ifndef USE_OPEN_BLAS
#define USE_INTEL_MKL
#else
#undef USE_INTEL_MKL
#endif
// *****************************************************************

#ifdef __cplusplus
extern "C" {
#endif
#ifdef USE_OPEN_BLAS
#include <cblas.h>
extern void openblas_set_num_threads(int numThreads);
#endif
#ifdef USE_INTEL_MKL
#include <mkl.h>
#include <mkl_service.h>
#endif
#ifdef __cplusplus
}
#endif

// This method is called during every systemml call, but the if condition is only executed once.
void ensureSequentialBLAS();

// Multiplies two matrices m1Ptr and m2Ptr in row-major format of shape
// (m1rlen, m1clen) and (m1clen, m2clen)
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen,
             int m1clen, int m2clen, int numThreads);
             
#endif
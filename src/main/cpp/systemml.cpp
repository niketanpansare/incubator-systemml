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
 

#include "systemml.h"
#include "libmatrixmult.h"
#include "libmatrixdnn.h"
#include <ctime>

// Linux:
// g++ -o libsystemml_mkl-linux-x86_64.so *.cpp  -I$JAVA_HOME/include -I$MKLROOT/include -I$JAVA_HOME/include/linux -lmkl_rt -lpthread  -lm -ldl -DUSE_INTEL_MKL -DUSE_GNU_THREADING -L$MKLROOT/lib/intel64 -m64 -fopenmp -O3 -shared -fPIC
// g++ -o libsystemml_openblas-linux-x86_64.so *.cpp  -I$JAVA_HOME/include  -I$JAVA_HOME/include/linux -lopenblas -lpthread -lm -ldl -DUSE_OPEN_BLAS -L/opt/OpenBLAS/lib/ -fopenmp -O3 -shared -fPIC

// Results from Matrix-vector/vector-matrix 1M x 1K, dense show that GetDoubleArrayElements creates a copy on OpenJDK.

// JNI Methods to get/release double* 
#define GET_DOUBLE_ARRAY(env, input) \
  ((double*)env->GetPrimitiveArrayCritical(input, NULL))
// env->GetDoubleArrayElements(input,NULL)
 
#define RELEASE_DOUBLE_ARRAY(env, input, inputPtr) \
  env->ReleasePrimitiveArrayCritical(input, inputPtr, 0)
// env->ReleaseDoubleArrayElements(input, inputPtr, 0)

double matmultTime = 0;

JNIEXPORT void JNICALL Java_org_apache_sysml_utils_NativeHelper_matrixMultDenseDense(
    JNIEnv* env, jclass cls, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret,
    jint m1rlen, jint m1clen, jint m2clen, jint numThreads) {
  double* m1Ptr = GET_DOUBLE_ARRAY(env, m1);
  double* m2Ptr = GET_DOUBLE_ARRAY(env, m2);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);

  clock_t t = clock();
  matmult(m1Ptr, m2Ptr, retPtr, (int)m1rlen, (int)m1clen, (int)m2clen, (int)numThreads);
  t = clock() - t;
  matmultTime += ((double)t * 1000)/CLOCKS_PER_SEC; // store in milliseconds

  RELEASE_DOUBLE_ARRAY(env, m1, m1Ptr);
  RELEASE_DOUBLE_ARRAY(env, m2, m2Ptr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);    
}


JNIEXPORT jdouble JNICALL Java_org_apache_sysml_utils_NativeHelper_getStatistics
  (JNIEnv * env, jclass cls, jint op) {
	// For now only return matmult execution time
	double tmp = matmultTime;
	matmultTime = 0;
	return (jdouble) tmp;
}

JNIEXPORT void JNICALL Java_org_apache_sysml_utils_NativeHelper_conv2dDense(
	JNIEnv* env, jclass, jdoubleArray input, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  
  conv2dBiasAddDense(inputPtr, 0, filterPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S,
    (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q, false, (int) numThreads);
    
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr); 
}

JNIEXPORT void JNICALL Java_org_apache_sysml_utils_NativeHelper_conv2dBiasAddDense(
	JNIEnv* env, jclass, jdoubleArray input, jdoubleArray bias, jdoubleArray filter,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
    
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* biasPtr = GET_DOUBLE_ARRAY(env, bias);
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  
  conv2dBiasAddDense(inputPtr, biasPtr, filterPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S,
    (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q, true, (int) numThreads);
    
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, bias, biasPtr);
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr); 
}

JNIEXPORT void JNICALL Java_org_apache_sysml_utils_NativeHelper_conv2dBackwardDataDense(
	JNIEnv* env, jclass, jdoubleArray filter, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  
  double* filterPtr = GET_DOUBLE_ARRAY(env, filter);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  
  conv2dBackwardDataDense(filterPtr, doutPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S,
    (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q, (int) numThreads);
  
  RELEASE_DOUBLE_ARRAY(env, filter, filterPtr);
  RELEASE_DOUBLE_ARRAY(env, dout, doutPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_utils_NativeHelper_conv2dBackwardFilterDense(
	JNIEnv* env, jclass, jdoubleArray input, jdoubleArray dout,
    jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S,
    jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q, jint numThreads) {
  double* inputPtr = GET_DOUBLE_ARRAY(env, input);
  double* doutPtr = GET_DOUBLE_ARRAY(env, dout);
  double* retPtr = GET_DOUBLE_ARRAY(env, ret);
  
  conv2dBackwardFilterDense(inputPtr, doutPtr, retPtr, (int) N, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S,
    (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q, (int) numThreads);
  
  RELEASE_DOUBLE_ARRAY(env, input, inputPtr);
  RELEASE_DOUBLE_ARRAY(env, dout, doutPtr);
  RELEASE_DOUBLE_ARRAY(env, ret, retPtr);
}
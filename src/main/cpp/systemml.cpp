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
 #include <vector>
 #include <cstdlib>
 #include <iostream>
 #include <cstdio>
 
 #ifdef __cplusplus
	extern "C"
	{
	#endif
	   #include <cblas.h>
	#ifdef __cplusplus
	}
 #endif
 
 enum SUPPORTED_BLOCKS { GENERIC, FOR }; 
 
/********************************************************************
1. Download community version of Intel MKL from https://software.intel.com/sites/campaigns/nest/

2. Compile systemml shared library
export MKL_ROOT=/opt/intel/mkl
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.el7_2.x86_64
g++ -shared -fPIC -o libsystemml.so systemml.cpp -I. -I$MKLROOT/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -fopenmp -L$MKL_ROOT/lib/intel64/ -lmkl_rt -lm

3. Make sure that MKL and systemml shared library are available to Java
export LD_LIBRARY_PATH=$MKL_ROOT/lib/intel64:.:$LD_LIBRARY_PATH
 ********************************************************************/
 // -------------------------------------
 // Objects available to the instructions
 double** denseBlocks;
 int* numRows;
 int* numCols;
 // -------------------------------------
 

 class Instruction {
 	public:
 	void execute() {
 		// TODO:
 	}
 };
 
 class ProgramBlock {
 	public:
 	std::vector<Instruction> instructions;
 	int type, arg1, arg2, arg3;
 	std::vector<ProgramBlock> childBlocks;
 	void execute() {
 		if(type == GENERIC) {
	 		for(int i = 0; i < instructions.size(); i++) {
	 			instructions[i].execute();
	 		}
 		}
 		else if(type == FOR) {
 			for(int i = arg1; i <= arg2; i += arg3) {
	 			for(int j = 0; j < childBlocks.size(); j++) {
	 				childBlocks[j].execute();
	 			}
 			}
 		}
 		else {
 			std::cout << "The block is not supported";
 			exit(0);
 		}
 	}
 };
 
 void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen, int m1clen, int m2clen) {
	int m = m1rlen;
	int n = m2clen;
	int k = m1clen;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, m1Ptr, k, m2Ptr, n, 0.0, retPtr, n);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_matrixMultDenseDense
 (JNIEnv * env, jclass cls, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret, jint m1rlen, jint m1clen, jint m2clen) {
	double* m1Ptr  = env->GetDoubleArrayElements(m1,NULL);
	double* m2Ptr  = env->GetDoubleArrayElements(m2,NULL);
	double* retPtr  = env->GetDoubleArrayElements(ret,NULL);

	matmult(m1Ptr, m2Ptr, retPtr, (int) m1rlen, (int) m1clen, (int) m2clen);
	env->ReleaseDoubleArrayElements(m1,m1Ptr, 0);
	env->ReleaseDoubleArrayElements(m2,m2Ptr, 0);
	env->ReleaseDoubleArrayElements(ret,retPtr, 0);
}
 
 // Setup the denseBlocks, parse encodedBlk and call execute()
 JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_controlprogram_CPPUtil_execute(JNIEnv * env, jobject obj, 
 	jintArray jencodedBlk, jint jlenEncodedBlock, jint jnumVarID) {
 	jclass cls = env->GetObjectClass(obj);
 	
 	jmethodID getDenseBlockMethodID = env->GetMethodID(cls, "getDenseBlock", "(I)[D");
 	jmethodID getNumRowsMethodID = env->GetMethodID(cls, "getNumRows", "(I)I");
 	jmethodID getNumColsMethodID = env->GetMethodID(cls, "getNumCols", "(I)I");
 	int lenEncodedBlock = (int) jlenEncodedBlock;
 	int numVarID = (int) jnumVarID;
 	jdoubleArray** arr = new jdoubleArray*[numVarID];
 	denseBlocks = new double*[numVarID];
 	numRows = new int[numVarID];
 	numCols = new int[numVarID];
 	
 	int* encodedBlk = reinterpret_cast<int*>(env->GetIntArrayElements(jencodedBlk, NULL));
 	
 	#pragma omp parallel for
 	for(int i = 0; i < numVarID; i++) {
 		// We prefer this approach rather than calling getDenseBlock per instruction
 		// Pros: One time penalty before the execution of the for loop
 		// Cons: Apriori memory requirement (can potentially result in OOM if called with smaller JVM)
 		// Get dense blocks
 		jobject ret = env->CallObjectMethod(obj, getDenseBlockMethodID, (jint) i); 
	 	arr[i] = reinterpret_cast<jdoubleArray*>(&ret);
	 	// TODO: This might involve copying. Investigate direct access via GetPrimitiveArrayCritical or NewDirectByteBuffer.
	 	denseBlocks[i] = env->GetDoubleArrayElements(*(arr[i]), NULL);
	 	
	 	// Get number of rows and columns
	 	numRows[i] = (int) env->CallIntMethod(obj, getNumRowsMethodID, (jint) i);
	 	numCols[i] = (int) env->CallIntMethod(obj, getNumColsMethodID, (jint) i);
 	}
 	
 	env->ReleaseIntArrayElements(jencodedBlk, encodedBlk, 0);
 	#pragma omp parallel for
 	for(int i = 0; i < numVarID; i++) {
 		// Release dense blocks
 		env->ReleaseDoubleArrayElements(*(arr[i]), denseBlocks[i], 0);
 	}
 	
 	// No need to delete the content
 	delete [] denseBlocks;
 	delete [] arr;
 	delete [] numRows;
 	delete [] numCols;
 }
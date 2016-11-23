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
 
 #include "SystemML.h"
 #include <vector>
 #include <cstdlib>
 #include <iostream>
 #include <cstdio>
 
 enum SUPPORTED_BLOCKS { GENERIC, FOR }; 
 
 // gcc -shared -I"C:\\Program Files\\Java\\jdk1.7.0_79\\include" -I"C:\\Program Files\\Java\\jdk1.7.0_79\\include\\win32" -o SystemML.dll SystemML.cpp
 // g++ -shared -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -fPIC -o SystemML.so SystemML.cpp
 // -------------------------------------
 // Objects available to the instructions
 double** denseBlocks;
 int* numRows;
 int* numCols;
 // -------------------------------------
 
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen, int m1clen, int m2clen) {
	// TODO: With MKL
}
 
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
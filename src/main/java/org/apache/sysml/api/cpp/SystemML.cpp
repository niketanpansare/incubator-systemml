#include <iostream>
#include "SystemML.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

// Currently no benefits observed
// #define AVOID_ZEROIN_IM2COL

#define BATCH_SIZE 256
#define MAX(a, b) ((a) < (b)) ? (b) : (a)

// Use following compilers
// GCC 4.2 and newer,
// MSVC 2008 and newer
#include <Eigen/Dense>
using namespace Eigen;
void matmult(double* m1Ptr, double* m2Ptr, double* retPtr, int m1rlen, int m1clen, int m2clen) {
	MatrixXd eigenM1 = Map<MatrixXd>( m2Ptr, (int)m2clen, (int)m1clen);
 	MatrixXd eigenM2 = Map<MatrixXd>( m1Ptr, (int)m1clen, (int)m1rlen);
 	
	MatrixXd eigenRet = eigenM1 * eigenM2;
	memcpy(retPtr, eigenRet.data(), sizeof(double)*(int)m1rlen*(int)m2clen);
}

void im2col(double* inputArray, double* outputArray, int N, int C, int H, int W, int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int P, int Q) {
	for (int n = 0; n < N; n++) { // Do following for all images
		for (int c = 0; c < C; c++) { // Since format is NCHW
			const int inputOffset = n*C*H*W + c*H*W;
			const int outputOffset = (c*R*S*N + n)*P*Q;
			for (int r = 0; r < R; r++) { // Get an input patch of size R X S
				for (int s = 0; s < S; s++) {
					int localIndex = outputOffset + ((r*S*N + s*N)*P*Q);
				
					int input_row = r - pad_h;
					// And copy it to outputArray[i] (taking care of padding & striding)
					for (int p = P; p > 0; p--) {
						if (input_row >= 0 && input_row < H) {
							int input_col = s - pad_w;
							for (int q = Q; q > 0; q--, localIndex++) {
								if (input_col >= 0 && input_col < W) {
									// Copy from [channel c, height input_row, width input_col]
									outputArray[localIndex] = inputArray[inputOffset + input_row*W + input_col];
								}
								#ifdef AVOID_ZEROIN_IM2COL
								else {
									outputArray[localIndex] = 0;
								}
								#endif
								input_col += stride_w;
							}
						} else {
							#ifdef AVOID_ZEROIN_IM2COL
							for(int i = localIndex; i < localIndex + Q; i++) {
								outputArray[localIndex] = 0;
							}
							#endif
							localIndex += Q;
						}
						input_row += stride_h;
					}
				}
			}
		}
	}
}

void reshape_col(double* inputArray, double* outputArray, int N, int K, int P, int Q) {
	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++)  {
			memcpy(outputArray + n*K*P*Q + k*P*Q, inputArray+ k*N*P*Q + n*P*Q, P*Q);
		}
	}
}

void rotate180(double* inputArray, double* outputArray, int N, int K, int P, int Q) {
	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			for (int p = 0; p < P; p++) {
				for (int q = 0; q < Q; q++) {
					outputArray[n*K*P*Q + p*Q*K + q*K + k] = inputArray[n*K*P*Q + k*P*Q + p*Q + q];
				}
			}

		}
	}
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_matrix_data_LibMatrixNative_matrixMultDenseDense(JNIEnv * env, jclass, jdoubleArray m1, jdoubleArray m2, jdoubleArray ret, jint m1rlen, jint m1clen, jint m2clen) {
	double* m1Ptr  = env->GetDoubleArrayElements(m1,NULL);
	double* m2Ptr  = env->GetDoubleArrayElements(m2,NULL);
	double* retPtr  = env->GetDoubleArrayElements(ret,NULL);

	matmult(m1Ptr, m2Ptr, retPtr, (int) m1rlen, (int) m1clen, (int) m2clen);
	env->ReleaseDoubleArrayElements(m1,m1Ptr, 0);
	env->ReleaseDoubleArrayElements(m2,m2Ptr, 0);
	env->ReleaseDoubleArrayElements(ret,retPtr, 0);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_matrix_data_LibMatrixNative_conv2dDense
  (JNIEnv * env, jclass, jdoubleArray input, jdoubleArray filter, jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S, 
	jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
	
	// Im2col
	double* inputPtr  = env->GetDoubleArrayElements(input,NULL);
	double* filterPtr  = env->GetDoubleArrayElements(filter, NULL);
	MatrixXd eigenM2 = Map<MatrixXd>( filterPtr , (int)C*(int)R*(int)S, (int)K);
	double* retPtr  = env->GetDoubleArrayElements(ret,NULL);
	
	int numIm2ColElem = (int) C * (int) R * (int) S * (int) P * (int) Q;
	#pragma omp parallel for
	for(int i = 0; i < N; i++) {
		// Step 1: Perform im2col
		double* loweredMat = new double[numIm2ColElem];
		#ifndef AVOID_ZEROIN_IM2COL
			memset(loweredMat, 0, numIm2ColElem*sizeof(double));
		#endif
		im2col(inputPtr + i*C*H*W, loweredMat, 1, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q);
		
		// Step 2: Matrix multiplication
		MatrixXd eigenM1 = Map<MatrixXd>( loweredMat, 1 * (int) P * (int) Q, (int)C*(int)R*(int)S);
		MatrixXd eigenRet = eigenM1 * eigenM2;
		
		// Step 3: reshape col
		reshape_col(eigenRet.data(), retPtr + i*K*P*Q, 1, (int)K, (int)P, (int)Q);
		delete [] loweredMat;
	}
	env->ReleaseDoubleArrayElements(input, inputPtr, 0);
	env->ReleaseDoubleArrayElements(filter, filterPtr, 0);
	env->ReleaseDoubleArrayElements(ret, retPtr, 0);
}

JNIEXPORT void JNICALL Java_org_apache_sysml_runtime_matrix_data_LibMatrixNative_conv2dBackwardFilterDense 
  (JNIEnv * env, jclass, jdoubleArray input, jdoubleArray dout, jdoubleArray ret, jint N, jint C, jint H, jint W, jint K, jint R, jint S, 
	jint stride_h, jint stride_w, jint pad_h, jint pad_w, jint P, jint Q) {
		
	double* inputPtr  = env->GetDoubleArrayElements(input, NULL);
	double* doutPtr  = env->GetDoubleArrayElements(dout, NULL);
	double* tempRet = new double[BATCH_SIZE*(int)K*(int)C*(int)R*(int)S];
	double* retPtr  = env->GetDoubleArrayElements(ret,NULL);
	
	memset(retPtr, 0, sizeof(double)*(int)K*(int)C*(int)R*(int)S);
	
	int numIm2ColElem = (int) C * (int) R * (int) S * (int) P * (int) Q;
	
	// Batching reduce the memory requirements for tempRet
	for(int j = 0; j < N; j += BATCH_SIZE) {
		int maxN = MAX(j+BATCH_SIZE, N);
		#pragma omp parallel for
		for(int i = j; i < N; i++) {
			// Step 1: Perform im2col
			double* loweredMat = new double[numIm2ColElem];
			#ifndef AVOID_ZEROIN_IM2COL
				memset(loweredMat, 0, numIm2ColElem*sizeof(double));
			#endif
			im2col(inputPtr + i*C*H*W, loweredMat, 1, (int) C, (int) H, (int) W, (int) K, (int) R, (int) S, (int) stride_h, (int) stride_w, (int) pad_h, (int) pad_w, (int) P, (int) Q);
			
			// Step 2: Perform rotate 180
			double* rotatedDout = new double[(int) K *(int) P * (int) Q];	
			rotate180(doutPtr + i*K*P*Q, rotatedDout, 1, (int)K, (int)P, (int)Q);
			
			// Step 3: Matrix multiplication
			MatrixXd eigenM1 = Map<MatrixXd>( rotatedDout, (int)K, (int)P*(int)Q);
			MatrixXd eigenM2 = Map<MatrixXd>( loweredMat, (int)P*(int)Q, (int)C*(int)R*(int)S);
			// -----------------------------------------
			// Most compute-intensive part
			MatrixXd eigenRet = eigenM1 * eigenM2;
			MatrixXd eigenRet1 = eigenRet.transpose();
			// -----------------------------------------
			memcpy(tempRet + i*(int)K*(int)C*(int)R*(int)S, eigenRet1.data(), sizeof(double)*(int)K*(int)C*(int)R*(int)S);
			
			delete [] loweredMat;
			delete [] rotatedDout;
		}
		
		#pragma omp parallel for
		for(int k = 0; k < K; k++) {
			for(int n = 0; n < maxN; n++) {
				for(int c = 0; c < C; c++) {
					for(int r = 0; r < R; r++) {
						for(int s = 0; s < S; s++) {
							retPtr[k*C*R*S + c*R*S + r*S + s] += tempRet[n*K*C*R*S + k*C*R*S + c*R*S + r*S + s];
						}
					}	
				}
			}	
		}
	}
	
	delete [] tempRet;
	env->ReleaseDoubleArrayElements(ret, retPtr, 0);
	env->ReleaseDoubleArrayElements(input, inputPtr, 0);
	env->ReleaseDoubleArrayElements(dout, doutPtr, 0);
}


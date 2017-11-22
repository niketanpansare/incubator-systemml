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
package org.apache.sysml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.concurrent.Callable;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNNIm2ColHelper.Im2colWorker;
import org.apache.sysml.utils.NativeHelper;

/**
 * This class contains the set of operators used for performing conv2d
 */
public class LibMatrixDNNConv2dHelper {

	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) = output.
	 * This operator has less memory pressure than LoopedIm2ColConv2dAllChannels.
	 */
	public static class LoopedIm2ColConv2dOneChan implements Callable<Long> 
	{
		protected final int _rl, _ru; 
		protected final ConvolutionParameters _params; 
		protected final ArrayList<MatrixBlock> _filters;
		
		public LoopedIm2ColConv2dOneChan(int rl, int ru, ConvolutionParameters params, ArrayList<MatrixBlock> filters) {
			_rl = rl; _ru = ru;
			_params = params; 
			_filters = filters;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K;
			int RS = _params.R*_params.S;
			MatrixBlock im2ColOutBlock = new MatrixBlock(RS, PQ, false);
			Im2colWorker im2ColWorker = Im2colWorker.getWorker( _params.input1, im2ColOutBlock, _params, false, false);
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++)  {
				for(int c = 0; c < _params.C; c++)  {
					// im2col(input) => _im2ColOutBlock
					long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
					im2ColWorker.execute(n, c);
					long t2 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
					
					// filter %*% _im2ColOutBlock => matMultOutBlock
					MatrixBlock matMultOutBlock = new MatrixBlock(K, PQ, false);
					LibMatrixDNNHelper.singleThreadedMatMult(_filters.get(c), im2ColOutBlock, matMultOutBlock, false, true, _params);
					long t3 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
					
					if(DMLScript.FINEGRAINED_STATISTICS) {
						time1 += t2 - t1;
						time2 += t3 - t2;
					}
					
					// Add the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
					add(matMultOutBlock, _params.output.getDenseBlock(), n*K*PQ, K, PQ);
				}
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					LibMatrixDNNHelper.addBias(n, _params.output.getDenseBlock(), _params.bias.getDenseBlock(), K, PQ);
			}
			if(DMLScript.FINEGRAINED_STATISTICS) {
				LibMatrixDNN.loopedConvIm2ColTime.addAndGet(time1);
				LibMatrixDNN.loopedConvMatMultTime.addAndGet(time2);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
		
		// Copy the matrix src of shape [K X PQ] to params.output.denseBlock + destPos
		private static void add(MatrixBlock src, double [] dest, int destPos, int K, int PQ) {
			// Copying is required as LibMatrixMult.matrixMult (and/or Java) is not pointer aware.
			// This is not required in Native implementation
			if(!src.isEmptyBlock()) {
				if(src.isInSparseFormat()) {
					// Copy the sparse matrix matMultOutBlock of shape [K X PQ] to 
					// params.output.denseBlock + destPos
					for(int k = 0; k < src.getNumRows(); k++) {
						if( !src.sparseBlock.isEmpty(k) ) {
							int apos = src.sparseBlock.pos(k);
							int alen = src.sparseBlock.size(k);
							int[] aix = src.sparseBlock.indexes(k);
							double[] avals = src.sparseBlock.values(k);
							int desPosK = destPos + k*PQ;
							for(int j = apos; j < apos+alen; j++) {
								int pqIndex = aix[j];
								dest[desPosK + pqIndex ] += avals[j];
							}
						}
					}
				}
				else {
					LibMatrixMult.vectAdd(src.denseBlock, dest, 0, destPos, K*PQ);
				}
			}
		}
	}	
	
	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) = output
	 */
	public static class LoopedIm2ColConv2dAllChan implements Callable<Long> 
	{
		protected final int _rl, _ru; 
		protected final ConvolutionParameters _params;
		
		public LoopedIm2ColConv2dAllChan(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			final int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock outIm2col = new MatrixBlock(CRS, PQ, false);
			MatrixBlock outMM = new MatrixBlock(K, PQ, false);
			Im2colWorker im2ColWorker = Im2colWorker.getWorker( _params.input1, outIm2col, _params, true, false);
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++)  {
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				im2ColWorker.execute(n);
				long t2 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				// filter %*% _im2ColOutBlock => matMultOutBlock
				outMM.reset(outMM.rlen, outMM.clen, false);
				LibMatrixDNNHelper.singleThreadedMatMult(_params.input2, outIm2col, outMM, false, true, _params);
				long t3 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				if(DMLScript.FINEGRAINED_STATISTICS) {
					time1 += t2 - t1;
					time2 += t3 - t2;
				}
				
				// Copy the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
				partialCopy1(outMM, _params.output.getDenseBlock(), n*K*PQ, K, PQ);
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					LibMatrixDNNHelper.addBias(n, _params.output.getDenseBlock(), _params.bias.getDenseBlock(), K, PQ);
			}
			
			if(DMLScript.FINEGRAINED_STATISTICS) {
				LibMatrixDNN.loopedConvIm2ColTime.addAndGet(time1);
				LibMatrixDNN.loopedConvMatMultTime.addAndGet(time2);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
		
		// Copy the matrix src of shape [K X PQ] to params.output.denseBlock + destPos
		private static void partialCopy1(MatrixBlock src, double [] dest, int destPos, int K, int PQ) {
			// Copying is required as LibMatrixMult.matrixMult (and/or Java) is not pointer aware.
			// This is not required in Native implementation
			if( src.isEmptyBlock() )
				return;
			if(src.isInSparseFormat()) {
				SparseBlock sblock = src.sparseBlock;
				for(int k = 0; k < src.getNumRows(); k++) {
					if( sblock.isEmpty(k) ) continue;
					int apos = sblock.pos(k);
					int alen = sblock.size(k);
					int[] aix = sblock.indexes(k);
					double[] avals = sblock.values(k);
					int desPosK = destPos + k*PQ;
					for(int j = apos; j < apos+alen; j++)
						dest[desPosK+aix[j]] = avals[j];
				}
			}
			else 
				System.arraycopy(src.denseBlock, 0, dest, destPos, K * PQ);
		}
	}
	
	/**
	 * This implementation is similar to LoopedIm2ColConv2dAllChan, except for using a 
	 * sparse-dense matrix multiplication with t(t(Xi) %*% t(F)) instead of a 
	 * dense-sparse matrix multiplication with Xi %*% F.
	 * 
	 * NOTE: this implementation assumes that the filter is passed in transposed form
	 * in order to share this temporary matrix (and its creation cost) across threads.
	 */
	public static class LoopedIm2ColConv2dTransAllChan extends LoopedIm2ColConv2dAllChan
	{
		public LoopedIm2ColConv2dTransAllChan(int rl, int ru, ConvolutionParameters params) {
			super(rl, ru, params);
		}

		@Override
		public Long call() throws Exception {
			final int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock outIm2col = new MatrixBlock(PQ, CRS, false);
			MatrixBlock outMM = new MatrixBlock(PQ, K, false);
			Im2colWorker im2ColWorker = Im2colWorker.getWorker( _params.input1, outIm2col, _params, true, true);
			
			for(int n = _rl; n < _ru; n++)  {
				// im2col(input) => _im2ColOutBlock
				im2ColWorker.execute(n);
				
				// t(_im2ColOutBlock) %*% t(filter) => t(matMultOutBlock)
				outMM.reset(outMM.rlen, outMM.clen, false);
				LibMatrixDNNHelper.singleThreadedMatMult(outIm2col, _params.input2, outMM, false, false, _params);
				
				// Copy the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
				partialCopyTrans(outMM, _params.output, n*K*PQ, K, PQ);
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					LibMatrixDNNHelper.addBias(n, _params.output.getDenseBlock(), _params.bias.getDenseBlock(), K, PQ);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
		
		private static void partialCopyTrans(MatrixBlock src, MatrixBlock dest, int destPos, int K, int PQ) {
			if( src.isEmptyBlock() )
				return;
			//copy src into its destination row w/ piggybacked transpose
			//src is [PQ x K] -> [K x PQ] -> [1 x KPQ]
			if(src.isInSparseFormat()) {
				SparseBlock sblock = src.sparseBlock;
				double[] c = dest.denseBlock;
				for(int i = 0; i < src.getNumRows(); i++) {
					if( sblock.isEmpty(i) ) continue;
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] avals = sblock.values(i);
					int desPosK = destPos + i;
					for(int j = apos; j < apos+alen; j++)
						c[desPosK+aix[j]*PQ] = avals[j];
				}
			}
			else {
				double[] a = src.denseBlock;
				double[] c = dest.denseBlock;
				final int blocksizeIJ = 128; //128KB for L2
				//cache-conscious blocked execution
				for( int bi = 0; bi < PQ; bi+=blocksizeIJ )
					for( int bj = 0; bj < K; bj+=blocksizeIJ ) {
						int bimin = Math.min(bi+blocksizeIJ, PQ);
						int bjmin = Math.min(bj+blocksizeIJ, K);
						//core transpose operation
						for(int i=bi, aix=bi*K+bj, cix=bj*PQ+bi; i<bimin; i++, aix+=K, cix++)
							LibMatrixReorg.transposeRow(a, c, aix, destPos+cix, PQ, bjmin-bj);
					}
			}
		}
	}
	
	/**
	 * This operator is used only for conv1d when input is sparse and filter is dense.
	 * 
	 * In the current version, we only support stride = 1 and pad = 0 for conv1d as these as the most common parameters.
	 */
	public static class SparseInputDenseFilterConv1dStride1Pad0AllChan implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; double [] _reshapeFilter;
		public SparseInputDenseFilterConv1dStride1Pad0AllChan(int rl, int ru, ConvolutionParameters params, double [] reshapeFilter) throws DMLRuntimeException {
			_rl = rl; _ru = ru;
			_params = params;
			_reshapeFilter = reshapeFilter;
			if(_params.Q != 1) {
				throw new DMLRuntimeException("Incorrect usage: Expected output width to be 1 but instead is " + _params.Q);
			}
		}
		
		public static boolean isApplicable(ConvolutionParameters params, int numThreads) {
			int numElemsInIntermediateMemory = numThreads*params.C*params.R*params.S*params.P*params.Q;
			return params.W == params.S && params.input1.isInSparseFormat() && !params.input2.isInSparseFormat() &&
					params.pad_h == 0 && params.pad_w == 0 && params.stride_h == 1 && params.stride_w == 1 &&
					// only use this operator if memory for reshaping the filter is accounted for by the optimizer to avoid OOM
					params.input2.getDenseBlock().length < numElemsInIntermediateMemory; 
		}
		
		public static double [] getReshapedFilter(ConvolutionParameters params) {
			double [] reshapeFilter = null;
			// reshape only if filter size is less than already accounted intermediate memory
			double [] filter = params.input2.denseBlock;
			reshapeFilter = new double[filter.length];
			// KCRS => CSKR
			int K = params.K; int C = params.C;
			int R = params.R; int S = params.S;
			int SKR = S*K*R; int KR = K*R;
			int i = 0;
			for(int k = 0; k < K; k++) {
				for(int c = 0; c < C; c++) {
					for(int r = 0; r < R; r++) {
						for(int s = 0; s < S; s++, i++) {
							reshapeFilter[c*SKR + s*KR + k*R + r] = filter[i];
						}
					}
				}
			}
			return reshapeFilter;
		}

		@Override
		public Long call() throws Exception {
			int KPQ = _params.K*_params.P*_params.Q;
			int PQ = _params.P*_params.Q; int P = _params.P; int Q = _params.Q;
			int HW = _params.H*_params.W;
			int W = _params.W; int K = _params.K;
			int R = _params.R;
			int SKR = _params.S*K*R; int KR = K*R;
			double [] output = _params.output.denseBlock;
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					int outOffset = n*KPQ;
					// Iterate over the sparse block
					for(int j=apos; j<apos+alen; j++) {
						// Note: the input is of shape [N, CHW]
						int chw = aix[j];
						// Get individual zero-based c,h,w indexes from zero-based 'chw'
						int c = chw / HW;
						int h = (chw - c*HW)/W;
						int w = chw % W;
						double val = avals[j];
						
						int rMin = Math.max(0, h-P+1);
						int rMax = Math.min(R-1, h);
						int filterOffset = c*SKR + w*KR;
						for(int k = 0; k < K; k++) {
							int filterIndex = filterOffset + k*R + rMin;
							int outOffset1 = outOffset + k*PQ + h*Q;
							for(int r = rMin; r <= rMax; r++) {
								output[outOffset1 - r*Q] += val * _reshapeFilter[filterIndex++];
							}
						}
					}
				}
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					LibMatrixDNNHelper.addBias(n, output, _params.bias.denseBlock, K, PQ);
			}
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * This operator is used only if native is enabled, filter is dense and input is sparse
	 */
	public static class SparseNativeConv2d implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params;
		public SparseNativeConv2d(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int KPQ = _params.K*_params.P*_params.Q;
			int PQ = _params.P*_params.Q; int K = _params.K;
			double[] temp = new double[KPQ];
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					NativeHelper.conv2dSparse(apos, alen, aix, avals, _params.input2.getDenseBlock(), temp, 
							1, _params.C, _params.H, _params.W, _params.K, _params.R, _params.S, 
							_params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
					System.arraycopy(temp, 0, _params.output.denseBlock, n*KPQ, KPQ);
				}
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					LibMatrixDNNHelper.addBias(n, _params.output.denseBlock, _params.bias.denseBlock, K, PQ);
			}
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
}

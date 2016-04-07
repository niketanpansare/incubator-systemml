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
package org.apache.sysml.runtime.instructions.gpu;

import java.lang.ref.SoftReference;
import java.util.ArrayList;

import jcuda.Pointer;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.JCudaContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;

import static jcuda.jcudnn.JCudnn.cudnnConvolutionForward;
import static jcuda.jcudnn.JCudnn.cudnnDestroyTensorDescriptor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import jcuda.jcudnn.cudnnFilterDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyFilterDescriptor;
import static jcuda.runtime.JCuda.cudaFree;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnCreateConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyConvolutionDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnSetConvolutionNdDescriptor;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.JCudnn.cudnnGetConvolutionForwardWorkspaceSize;


// TODO: Using the same hierarchy as CPInstruction for now.
public class ConvolutionGPUInstruction extends UnaryCPInstruction {
	
	private CPOperand _in2; 
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();
	
	public ConvolutionGPUInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
		_cptype = CPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}
	
	public static ConvolutionGPUInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("conv2d")) {
			InstructionUtils.checkNumFields(parts, 15);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			out.split(parts[15]);
	
			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[3]));
			stride.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));
	
			return new ConvolutionGPUInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionCPInstruction: " + str);
		}
	}

	int N; int C; int H; int W;
	int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
	int P; int Q;
	
	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException {
		MatrixBlock outputBlock = null;
		if (instOpcode.equalsIgnoreCase("conv2d")) {
			
			pad_h = getScalarInput(ec, _padding, 0);
			pad_w = getScalarInput(ec, _padding, 1);
			stride_h = getScalarInput(ec, _stride, 0);
			stride_w = getScalarInput(ec, _stride, 1);

			N = getScalarInput(ec, _input_shape, 0);
			C = getScalarInput(ec, _input_shape, 1);
			H = getScalarInput(ec, _input_shape, 2);
			W = getScalarInput(ec, _input_shape, 3);

			K = getScalarInput(ec, _filter_shape, 0);
			
			R = getScalarInput(ec, _filter_shape, 2);
			S = getScalarInput(ec, _filter_shape, 3);
			
			P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
			Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
			
			if (instOpcode.equalsIgnoreCase("conv2d")) {
				MatrixBlock image = ec.getMatrixInput(input1.getName());
				MatrixBlock filter = ec.getMatrixInput(_in2.getName());
				if(image.isInSparseFormat() || filter.isInSparseFormat()) {
					throw new DMLRuntimeException("Sparse convolution not implemented");
				}
				double [] imageData = image.getDenseBlock();
				double [] filterData = filter.getDenseBlock();
				
				if(ec.gpuCtx != null) {
					JCudaContext gpuCtx = (JCudaContext) ec.gpuCtx;
					cudnnTensorDescriptor srcTensorDesc = null;
					cudnnTensorDescriptor dstTensorDesc = null;
					cudnnFilterDescriptor filterDesc = null;
					Pointer imagePointer = null;
					Pointer filterPointer = null;
					Pointer dstPointer = null;
					cudnnConvolutionDescriptor convDesc = null;
					Pointer workSpace = null;
					long sizeInBytes = 0;
					try {
						// Allocate descriptors
						srcTensorDesc = gpuCtx.allocateTensorDescriptor(N, C, H, W);
						dstTensorDesc = gpuCtx.allocateTensorDescriptor(N, K, P, Q);
						filterDesc = gpuCtx.allocateFilterDescriptor(K, C, R, S);
						
						// Allocate data
						imagePointer = gpuCtx.allocateDoubleArrayOnGPU(imageData, N, C, H, W);
						filterPointer = gpuCtx.allocateDoubleArrayOnGPU(filterData, K, C, R, S);
						dstPointer = gpuCtx.allocateDoubleArrayOnGPU(filterData, N, K, P, Q);
						
						convDesc = allocateConvolutionDescriptor();
						
						int algo = getAlgorithm();
						
						long sizeInBytesArray[] = { 0 };
			            workSpace = new Pointer();
			            cudnnGetConvolutionForwardWorkspaceSize(gpuCtx.cudnnHandle, 
			                    srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 
			                    algo, sizeInBytesArray);
			            
						Pointer alpha = gpuCtx.pointerTo(1.0); // TODO
						Pointer beta = gpuCtx.pointerTo(0.0f);
						int ret = cudnnConvolutionForward(gpuCtx.cudnnHandle, alpha, 
								srcTensorDesc, imagePointer, 
								filterDesc, filterPointer,
								convDesc, algo, workSpace, sizeInBytes, beta,
								dstTensorDesc, dstPointer);
						String status = jcuda.jcudnn.cudnnStatus.stringFor(ret);
						if(!status.equalsIgnoreCase("CUDNN_STATUS_SUCCESS")) {
							throw new DMLRuntimeException("Could not executed cudnnConvolutionForward: " + status);
						}
						
						outputBlock = allocateReusableNonZeroedDenseOutputBlock(ec, N, K * P * Q);
						gpuCtx.getDoubleArrayFromGPU(dstPointer, outputBlock.getDenseBlock());
					}
					finally {
						if(srcTensorDesc != null)
							cudnnDestroyTensorDescriptor(srcTensorDesc);
						if(dstTensorDesc != null)
							cudnnDestroyTensorDescriptor(dstTensorDesc);
						if(filterDesc != null)
							cudnnDestroyFilterDescriptor(filterDesc);
						if(imagePointer != null)
							cudaFree(imagePointer);
						if(dstPointer != null)
							cudaFree(dstPointer);
						if(filterPointer != null)
							cudaFree(filterPointer);
						if(convDesc != null)
							cudnnDestroyConvolutionDescriptor(convDesc);
						if(workSpace != null && sizeInBytes != 0)
							cudaFree(workSpace);
					}
				}
				else {
					throw new DMLRuntimeException("Unsupported GPU context for " + instOpcode);
				}
				
			}
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(_in2.getName());
		ec.setMatrixOutput(output.getName(), outputBlock);
	}
	
	cudnnConvolutionDescriptor allocateConvolutionDescriptor() {
		cudnnConvolutionDescriptor convDesc = new cudnnConvolutionDescriptor();
		cudnnCreateConvolutionDescriptor(convDesc);
		int padding [] = { pad_h, pad_w }; 
		int strides [] = { stride_h, stride_w };
		int upscale[] = { 1, 1 };
		cudnnSetConvolutionNdDescriptor(convDesc, 2, padding, strides, upscale, 
				CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
		return convDesc;
	}
	
	int getAlgorithm() {
		// TODO: For now always return GEMM, later optimize is
		return jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	}
	
	private SoftReference<double[]> reuseableNonZeroedDoubleArray;
	private MatrixBlock allocateReusableNonZeroedDenseOutputBlock(ExecutionContext ec, int numRowsOutput, int numColsOutput) throws DMLRuntimeException {
		long nnz = numRowsOutput * numColsOutput;
		MatrixBlock outputBlock = new MatrixBlock(numRowsOutput, numColsOutput, nnz);
		double[] outputArray = null;
		if(reuseableNonZeroedDoubleArray != null) {
			double[] arr = reuseableNonZeroedDoubleArray.get();
			if(arr != null && arr.length == nnz) {
				outputBlock.setDenseBlock(arr);
				outputArray = arr;
			}
		}
		
		if(outputArray == null) {
			outputBlock.allocateDenseBlock();
			outputArray = outputBlock.getDenseBlock();
			reuseableNonZeroedDoubleArray = new SoftReference<double[]>(outputArray);
		}
		
		outputBlock.setNonZeros(nnz);
		return outputBlock;
	}
}

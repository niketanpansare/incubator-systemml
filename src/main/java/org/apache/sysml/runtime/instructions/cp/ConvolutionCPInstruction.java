package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.ImageLayout;
import org.apache.sysml.api.DMLScript.TensorLayout;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class ConvolutionCPInstruction extends UnaryCPInstruction {
	
	private static final Log LOG = LogFactory.getLog(ConvolutionCPInstruction.class.getName());

	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();

	public ConvolutionCPInstruction(CPOperand in, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public static ConvolutionCPInstruction parseInstruction(String str)
			throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("reshape_col")
				|| opcode.equalsIgnoreCase("im2col")) {
			InstructionUtils.checkNumFields(parts, 14);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4,
			in.split(parts[1]);
			out.split(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
			stride.add(new CPOperand(parts[2]));
			stride.add(new CPOperand(parts[3]));
			padding.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			input_shape.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			filter_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));

			return new ConvolutionCPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}

	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// acquire inputs
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
		MatrixBlock outputBlock = null;
		if (instOpcode.equalsIgnoreCase("im2col")
				|| instOpcode.equalsIgnoreCase("reshape_col")) {
			pad_h = getScalarInput(ec, _padding, 0);
			pad_w = getScalarInput(ec, _padding, 1);
			stride_h = getScalarInput(ec, _stride, 0);
			stride_w = getScalarInput(ec, _stride, 1);

			if (DMLScript.imageLayout == ImageLayout.NCHW) {
				N = getScalarInput(ec, _input_shape, 0);
				C = getScalarInput(ec, _input_shape, 1);
				H = getScalarInput(ec, _input_shape, 2);
				W = getScalarInput(ec, _input_shape, 3);

				K = getScalarInput(ec, _filter_shape, 0);
				if (getScalarInput(ec, _filter_shape, 1) != C) {
					throw new DMLRuntimeException("The number of channels of input and filter should match");
				}
				R = getScalarInput(ec, _filter_shape, 2);
				S = getScalarInput(ec, _filter_shape, 3);
			} else if (DMLScript.imageLayout == ImageLayout.NHWC) {
				N = getScalarInput(ec, _input_shape, 0);
				H = getScalarInput(ec, _input_shape, 1);
				W = getScalarInput(ec, _input_shape, 2);
				C = getScalarInput(ec, _input_shape, 3);

				K = getScalarInput(ec, _filter_shape, 0);
				R = getScalarInput(ec, _filter_shape, 1);
				S = getScalarInput(ec, _filter_shape, 2);
				if (getScalarInput(ec, _filter_shape, 3) != C) {
					throw new DMLRuntimeException("The number of channels of input and filter should match");
				}
			}
			
			if (instOpcode.equalsIgnoreCase("im2col")) {
				checkInputDimension(matBlock);
				outputBlock = im2col(matBlock);
			} else {
				outputBlock = reshape_col(matBlock);
			}
		} else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}

		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), outputBlock);
	}
	
	int N; int C; int H; int W;
	int K; int R; int S; int stride_h; int stride_w; int pad_h; int pad_w;
	int P; int Q;
	double[] outputArray; double[] inputArray; MatrixBlock input;
	long outNNZ = 0;
	// These are used to iterate
	int i;  int row;

	
	private MatrixBlock init(int numRowsOutput) throws DMLRuntimeException {
		P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		
		if((W + 2 * pad_w - S) % stride_w != 0) {
			throw new DMLRuntimeException("The width does not work (Hint: (W + 2 * pad_w - S) % stride_w should be 0 [ ==> (" + W + "+" + " 2*" + pad_w + "-" +  S + ") % " + stride_w + "!= 0] ");
		}
		if((H + 2 * pad_h - R) % stride_h != 0) {
			throw new DMLRuntimeException("The height does not work (Hint: (H + 2 * pad_h - R) % stride_h should be 0 [ ==> (" + H + "+" + " 2*" + pad_h + "-" +  R + ") % " + stride_h + "!= 0] ");
		}
		
		int numColsOutput = N * P * Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		
		i = 0; row = 0; 
		return output;
	}
	
	// Reshape to 4D tensor (N, K, P, Q)
	private MatrixBlock reshape_col(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		
		this.input = input;
		P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		int numRowsOutput = N;
		int numColsOutput = K * P * Q;
		MatrixBlock output = new MatrixBlock(numRowsOutput, numColsOutput, numRowsOutput * numColsOutput);
		output.allocateDenseBlock();
		outputArray = output.getDenseBlock();
		output.setNonZeros(numRowsOutput * numColsOutput);
		inputArray = null;

		if (!input.isInSparseFormat())
			inputArray = input.getDenseBlock();
		i = 0; row = 0; 
		
		if(input.getNumColumns() != N*P*Q || input.getNumRows() != K) {
			throw new DMLRuntimeException("Incorrect input dimensions in col2im:" + input.getNumRows() + " " + input.getNumColumns());
		}
		
		if (DMLScript.imageLayout != ImageLayout.NCHW) {
			throw new DMLRuntimeException("col2im is not implemented for layout:" + DMLScript.imageLayout);
		}

		if (DMLScript.imageLayout == ImageLayout.NCHW) {
			for (int n = 0; n < N; n++) { 
				for (int k = 0; k < K; k++) { 
					for (int p = 0; p < P; p++) { 
						for (int q = 0; q < Q; q++) {
							if (DMLScript.tensorLayout == TensorLayout.W_XYZ) {
								if(inputArray != null)
									outputArray[n*K*P*Q + k*P*Q + p*Q + q] = inputArray[k*N*P*Q + n*P*Q + p*Q + q];
								else
									outputArray[n*K*P*Q + k*P*Q + p*Q + q] = input.getValue(k, n*P*Q + p*Q + q);
							}
						}
					}
				}
			}
		}

		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for col2im:" + execTime + " seconds.");
		return output;
	}
		

	private MatrixBlock im2col(MatrixBlock input) throws DMLRuntimeException {
		long start = System.nanoTime();
		
		this.input = input;
		MatrixBlock output = init(C * R * S);
		
		if (DMLScript.imageLayout != ImageLayout.NCHW) {
			throw new DMLRuntimeException("im2col is not implemented for layout:" + DMLScript.imageLayout);
		}

		if (DMLScript.imageLayout == ImageLayout.NCHW) {

			for (int c = 0; c < C; c++) { // Since format is NCHW
				for (int r = 0; r < R; r++) { // Get an input patch of size R X S
					for (int s = 0; s < S; s++) {
						for (int n = 0; n < N; n++) { // Do following for all images
							doIm2colOverInputPath_NCHW(n, c, r, s);
						}
					}
				}
			}
		}

		double execTime = (System.nanoTime() - start) / 1000000000;
		if (DMLScript.DEBUG_TENSOR && execTime > 5)
			LOG.info("Time for im2col:" + execTime + " seconds.");
		return output;
	}
	
	private void doIm2colOverInputPath_NCHW(int n, int c, int r, int s) {
		int input_row = r - pad_h;
		// And copy it to outputArray[i] (taking care of padding & striding)
		for (int p = P; p > 0; p--) {
			if (input_row >= 0 && input_row < H) {
				int input_col = s - pad_w;
				for (int q = Q; q > 0; q--, i++) {
					if (input_col >= 0 && input_col < W) {
						// Copy from [channel c, height input_row, width input_col]
						if (DMLScript.tensorLayout == TensorLayout.W_XYZ) {
							if (inputArray != null)
								outputArray[i] = inputArray[n*C*H*W + c*H*W + input_row*W + input_col];
							else
								outputArray[i] = input.getValue(n + 1, c*H*W + input_row*W + input_col + 1);
						} else if (DMLScript.tensorLayout == TensorLayout.WXY_Z) {
							outputArray[i] = input
									.getValue(n + 1, c*H*W + input_row*W + input_col + 1);
						}
					}
					input_col += stride_w;
				}
			} else {
				i += Q;
			}
			input_row += stride_h;
		}
	}

	
	
	private void checkInputDimension(MatrixBlock matBlock) throws DMLRuntimeException {
		if(DMLScript.tensorLayout == TensorLayout.W_XYZ && DMLScript.imageLayout == ImageLayout.NCHW 
				&& (N != matBlock.getNumRows() || C*H*W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
		else if(DMLScript.tensorLayout == TensorLayout.WXY_Z  && DMLScript.imageLayout == ImageLayout.NCHW 
				&& (N*C*H != matBlock.getNumRows() || W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
		if(DMLScript.tensorLayout == TensorLayout.W_XYZ && DMLScript.imageLayout == ImageLayout.NHWC 
				&& (N != matBlock.getNumRows() || C*H*W != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
		else if(DMLScript.tensorLayout == TensorLayout.WXY_Z  && DMLScript.imageLayout == ImageLayout.NHWC 
				&& (N*H*W != matBlock.getNumRows() || C != matBlock.getNumColumns())) {
			throw new DMLRuntimeException("Incorrect input shape in conv2d");
		}
	}
}

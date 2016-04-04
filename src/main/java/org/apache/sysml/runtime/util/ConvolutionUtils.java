package org.apache.sysml.runtime.util;

public class ConvolutionUtils {
	
	
	public static long getP(long H, long R, long verticalStride, long heightPadding) {
		long ret = (H + 2 * heightPadding - R) / verticalStride + 1;
		if(ret <= 0) {
			throw new RuntimeException("Incorrect output patch size: (image_height + 2 * pad_h - filter_height) / verticalStride + 1) needs to be positive, but is " + ret);
		}
		return ret;
		// return (long) Math.ceil( ( H - R + 1 + 2*heightPadding ) / verticalStride);
	}
	public static long getQ(long W, long S, long horizontalStride, long widthPadding) {
		long ret = (W + 2 * widthPadding - S) / horizontalStride + 1;
		if(ret <= 0) {
			throw new RuntimeException("Incorrect output patch size: (image_width + 2 * pad_w - filter_width) / horizontalStride + 1) needs to be positive, but is " + ret);
		}
		return ret;
		// return (long) Math.ceil( ( W - S + 1 + 2*widthPadding ) / horizontalStride);
	}
	
	// No performance benefits observed !!!
//	// Ideally this is suppose to be ConcurrentHashMap<Long, ArrayList<SoftReference<double[]>>>,
//	// but for first implementation, we will skip this
//	static ConcurrentHashMap<Integer, SoftReference<double[]>> reusableDoubleArrays = new ConcurrentHashMap<Integer, SoftReference<double[]>>();
//	public static void addDoubleArray(double[] arr) {
//		if(DMLScript.REUSE_NONZEROED_OUTPUT)
//			reusableDoubleArrays.putIfAbsent(arr.length, new SoftReference<double[]>(arr));
//	}
//	public static MatrixBlock allocateReusableNonZeroedDenseOutputBlock(ExecutionContext ec, int numRowsOutput, int numColsOutput)
//			throws DMLRuntimeException {
//		long length = numRowsOutput * numColsOutput;
//		long nnz = length;
//		MatrixBlock outputBlock = new MatrixBlock(numRowsOutput, numColsOutput, nnz);
//		boolean allocate = true;
//		if(DMLScript.REUSE_NONZEROED_OUTPUT) {
//			SoftReference<double[]> ref = reusableDoubleArrays.remove(length);
//			if(ref != null) {
//				double [] arr = ref.get();
//				if(arr != null) {
//					outputBlock.setDenseBlock(arr);
//					allocate = false;
//				}
//			}
//		}
//		if(allocate) {
//			outputBlock.allocateDenseBlock();
//		}
//		outputBlock.setNonZeros(nnz);
//		return outputBlock;
//	}
	
}

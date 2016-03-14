package org.apache.sysml.runtime.util;

public class ConvolutionUtils {
	
	
	public static long getP(long H, long R, long verticalStride, long heightPadding) {
		return (H + 2 * heightPadding - R) / verticalStride + 1;
		// return (long) Math.ceil( ( H - R + 1 + 2*heightPadding ) / verticalStride);
	}
	public static long getQ(long W, long S, long horizontalStride, long widthPadding) {
		return (W + 2 * widthPadding - S) / horizontalStride + 1;
		// return (long) Math.ceil( ( W - S + 1 + 2*widthPadding ) / horizontalStride);
	}
	
}

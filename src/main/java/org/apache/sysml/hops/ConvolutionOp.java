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

package org.apache.sysml.hops;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.ImageLayout;
import org.apache.sysml.api.DMLScript.TensorLayout;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.ConvolutionTransform;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.util.ConvolutionUtils;

public class ConvolutionOp extends Hop 
{
	public static boolean FORCE_DIST_SORT_INDEXES = false;
	
	public boolean bSortSPRewriteApplicable = false;
	
	private Hop.ConvOp op;

	private ConvolutionOp() {
		//default constructor for clone
	}
	
	public ConvolutionOp(String l, DataType dt, ValueType vt, ConvOp o, Hop inp)
	{
		super(l, dt, vt);
		op = o;
		getInput().add(0, inp);
		inp.getParent().add(this);
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}
	
	
	public ConvolutionOp(String l, DataType dt, ValueType vt, ConvOp o, ArrayList<Hop> inp) 
	{
		super(l, dt, vt);
		op = o;
		
		for( int i=0; i<inp.size(); i++ ) {
			Hop in = inp.get(i);
			getInput().add(i, in);
			in.getParent().add(this);
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public ConvOp getOp()
	{
		return op;
	}
	
	@Override
	public String getOpString() {
		String s = new String("");
		s += "r(" + HopsTransf2String.get(op) + ")";
		return s;
	}

	@Override
	public Lop constructLops()
		throws HopsException, LopsException 
	{
		//return already created lops
		if( getLops() != null )
			return getLops();

		ExecType et = optFindExecType();
		
		switch( op )
		{
			case IM2COL:
			case RESHAPE_COL:
			{
				if( et==ExecType.CP || et == ExecType.SPARK)
				{
					setLops(constructIm2colOrReshapeColLOP(et));
					break;
				}
				else {
					throw new HopsException("Unimplemented col2im/im2col for execution type: " + et.name());
				}
				// break;
			}
			default: 
				throw new HopsException("Unsupported lops construction for operation type '"+op+"'.");
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
				
		return getLops();
	}
	
	private Lop constructIm2colOrReshapeColLOP(ExecType et) throws HopsException, LopsException {
		if(getInput().size() != 13) {
			throw new HopsException("Incorrect number of inputs for im2col/col2im");
		}
		
		Lop in = getInput().get(0).constructLops();
		ConvolutionTransform transform1 = new ConvolutionTransform( in, 
				HopsConv2Lops.get(op), getDataType(), getValueType(), et);
		setOutputDimensions(transform1);
		setLineNumbers(transform1);
		
		// stride1, stride2, padding1, padding2  
		// input_shape1, input_shape2, input_shape3, input_shape4, 
		// filter_shape1, filter_shape2, filter_shape3, filter_shape4
		for( int i=1; i<=12; i++ )
		{
			Lop ltmp = getInput().get(i).constructLops();
			transform1.addInput(ltmp);
			ltmp.addOutput(transform1);
		}
		transform1.setLevel(); //force order of added lops
		return transform1;
	}

			
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		//no dedicated mem estimation per op type, because always propagated via refreshSizeInformation
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{	
		//default: no intermediate memory requirements
		return 0;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		long[] ret = null;
	
		Hop input = getInput().get(0);
		MatrixCharacteristics mc = memo.getAllInputStats(input);
			
		switch(op) 
		{
//			case DIAG:
//			{
//				// NOTE: diag is overloaded according to the number of columns of the input
//				
//				long k = mc.getRows(); 
//				
//				// CASE a) DIAG V2M
//				// input is a [1,k] or [k,1] matrix, and output is [k,k] matrix
//				// #nnz in output is in the worst case k => sparsity = 1/k
//				if( k == 1 )
//					ret = new long[]{k, k, ((mc.getNonZeros()>=0) ? mc.getNonZeros() : k)};
//				
//				// CASE b) DIAG M2V
//				// input is [k,k] matrix and output is [k,1] matrix
//				// #nnz in the output is likely to be k (a dense matrix)		
//				if( k > 1 )
//					ret = new long[]{k, 1, ((mc.getNonZeros()>=0) ? Math.min(k,mc.getNonZeros()) : k) };
//				
//				break;		
//			}
			case IM2COL:
			case RESHAPE_COL:
				// TODO:
		}	
		
		return ret;
	}
	

	@Override
	public boolean allowsAllExecTypes()
	{
		return true;
	}
	
	@Override
	protected ExecType optFindExecType() throws HopsException {
		
		checkAndSetForcedPlatform();
	
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;
		}
		else 
		{	
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			// Choose CP, if the input dimensions are below threshold or if the input is a vector
			else if ( getInput().get(0).areDimsBelowThreshold() || getInput().get(0).isVector() )
			{
				_etype = ExecType.CP;
			}
			else 
			{
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
	
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		Hop input1 = getInput().get(0);
		
		switch(op) 
		{
			case IM2COL:
			{
				try {
					// stride1, stride2, padding1, padding2  
					// input_shape1, input_shape2, input_shape3, input_shape4, 
					// filter_shape1, filter_shape2, filter_shape3, filter_shape4
					long stride1 = extractValue(getInput().get(1));
					long stride2 = extractValue(getInput().get(2));
					long padding1 = extractValue(getInput().get(3));
					long padding2 = extractValue(getInput().get(4));
					long N = -1; long C = -1; long H = -1; long W = -1;
					long K = -1; long R = -1; long S = -1;
					
					if(DMLScript.imageLayout == ImageLayout.NCHW) {
						N = extractValue(getInput().get(5));
						C = extractValue(getInput().get(6));
						H = extractValue(getInput().get(7));
						W = extractValue(getInput().get(8));
						K = extractValue(getInput().get(9));
						C = (C <= 0) ? extractValue(getInput().get(10)) : C;
						R = extractValue(getInput().get(11));
						S = extractValue(getInput().get(12));
					}
					else if(DMLScript.imageLayout == ImageLayout.NHWC) {
						N = extractValue(getInput().get(5));
						H = extractValue(getInput().get(6));
						W = extractValue(getInput().get(7));
						C = extractValue(getInput().get(8));
						K = extractValue(getInput().get(9));
						R = extractValue(getInput().get(10));
						S = extractValue(getInput().get(11));
						C = (C <= 0) ? extractValue(getInput().get(12)) : C;
					}
					
					// Set _dim1, _dim2 and if possible _nnz (use input1.getNnz())
					_dim1 = C*R*S;
					long P = ConvolutionUtils.getP(H, R, stride1, padding1);
					long Q = ConvolutionUtils.getQ(W, S, stride2, padding2);
					_dim2 = N*P*Q;
					if(input1.getNnz() >= 0) {
						// long approxNumPaddedZeros = N*C*(2*(P*R + Q*S));
						long numZerosInOriginalImage = (N*C*H*W - input1.getNnz());
						// long conservativeEstNumZeros = (numZerosInOriginalImage + approxNumPaddedZeros);
						// Worst-case estimates (assuming only nnz are replicated):
						// TODO:
						_nnz = _dim1*_dim2 - numZerosInOriginalImage; 
					}
				}
				catch(DMLRuntimeException e) {}
				
				break;
			}
			case RESHAPE_COL:
			{
				try {
					// stride1, stride2, padding1, padding2  
					// input_shape1, input_shape2, input_shape3, input_shape4, 
					// filter_shape1, filter_shape2, filter_shape3, filter_shape4
					long stride1 = extractValue(getInput().get(1));
					long stride2 = extractValue(getInput().get(2));
					long padding1 = extractValue(getInput().get(3));
					long padding2 = extractValue(getInput().get(4));
					long N = -1; long C = -1; long H = -1; long W = -1;
					long K = -1; long R = -1; long S = -1;
					
					// Set _dim1, _dim2 and if possible _nnz (use input1.getNnz())
					if(DMLScript.imageLayout == ImageLayout.NCHW) {
						N = extractValue(getInput().get(5));
						C = extractValue(getInput().get(6));
						H = extractValue(getInput().get(7));
						W = extractValue(getInput().get(8));
						K = extractValue(getInput().get(9));
						C = (C <= 0) ? extractValue(getInput().get(10)) : C;
						R = extractValue(getInput().get(11));
						S = extractValue(getInput().get(12));
					}
					else if(DMLScript.imageLayout == ImageLayout.NHWC) {
						N = extractValue(getInput().get(5));
						H = extractValue(getInput().get(6));
						W = extractValue(getInput().get(7));
						C = extractValue(getInput().get(8));
						K = extractValue(getInput().get(9));
						R = extractValue(getInput().get(10));
						S = extractValue(getInput().get(11));
						C = (C <= 0) ? extractValue(getInput().get(12)) : C;
					}
					
					long P = ConvolutionUtils.getP(H, R, stride1, padding1);
					long Q = ConvolutionUtils.getQ(W, S, stride2, padding2);
					if(DMLScript.tensorLayout == TensorLayout.W_XYZ) {
						_dim1 = N;
						_dim2 = K * P * Q;
						// TODO: nnz
						_nnz = _dim1*_dim2;
					}
					else if(DMLScript.tensorLayout == TensorLayout.WXY_Z && DMLScript.imageLayout == ImageLayout.NCHW) {
						_dim1 = N * K * P;
						_dim2 = Q;
						// TODO: nnz
						_nnz = _dim1*_dim2;
					}
					else if(DMLScript.tensorLayout == TensorLayout.WXY_Z && DMLScript.imageLayout == ImageLayout.NHWC) {
						_dim1 = N * P * Q;
						_dim2 = K;
						// TODO: nnz
						_nnz = _dim1*_dim2;
					}
					else {
						System.out.println("Incorrect layout");
					}
				}
				catch(DMLRuntimeException e) {}
				break;
			}
		}	
	}
	
	private long extractValue(Hop hop) throws DMLRuntimeException {
		if(hop instanceof LiteralOp)
			return (long) HopRewriteUtils.getDoubleValueSafe((LiteralOp)hop);
		throw new DMLRuntimeException("Cannot extract value");
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ConvolutionOp ret = new ConvolutionOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof ConvolutionOp) )
			return false;
		
		ConvolutionOp that2 = (ConvolutionOp)that;		
		boolean ret =  (op == that2.op)
				    && (getInput().size()==that.getInput().size());
				
		//compare all childs (see reshape, sort)
		if( ret ) //sizes matched
			for( int i=0; i<_input.size(); i++ )
				ret &= getInput().get(i) == that2.getInput().get(i);
		
		return ret;
	}
	
	
	@Override
	public void printMe() throws HopsException 
	{
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug("  Operation: " + op);
				for (Hop h : getInput()) {
					h.printMe();
				}
			}
			setVisited(VisitStatus.DONE);
		}
	}
}
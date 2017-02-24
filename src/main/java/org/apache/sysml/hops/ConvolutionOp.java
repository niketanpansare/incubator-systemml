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
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.lops.ConvolutionTransform;
import org.apache.sysml.lops.ConvolutionTransform.OperationTypes;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;

public class ConvolutionOp extends Hop  implements MultiThreadedHop
{	
	public int MAPSIDE_THRESHOLD = ConfigurationManager.getBlocksize();
	
	private Hop.ConvOp op;

	private int _maxNumThreads = -1; //-1 for unlimited

	private ConvolutionOp() {
		//default constructor for clone
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
		return "" + HopsConv2Lops.get(op);
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
			case MAX_POOLING:
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D_BACKWARD_DATA:
			case DIRECT_CONV2D_BACKWARD_FILTER:
			case BIAS_ADD:
			{	
				if(et == ExecType.CP || et == ExecType.GPU || et == ExecType.SPARK) {
					setLops(constructConvolutionLops(et));
					break;
				}
				else {
					throw new HopsException("Unimplemented ConvolutionOp for execution type: " + et.name());
				}
				// break;
			}
			case IM2COL:
			case RESHAPE_COL:
			case ROTATE180:
			case COL2IM:
			{	
				// Spark-specific operators
				setLops(constructConvolutionLops(ExecType.SPARK));
				break;
			}
			default: 
				throw new HopsException("Unsupported lops construction for operation type '"+op+"'.");
		}
		
		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
				
		return getLops();
	}
	
	public void setOp(ConvOp op) {
		this.op = op;
	}
	
	private int getNumExpectedInputs() {
		switch(op) {
			case MAX_POOLING_BACKWARD:
			case DIRECT_CONV2D_BACKWARD_FILTER:
			case DIRECT_CONV2D_BACKWARD_DATA:
				return 14;
			case BIAS_ADD:
				return 2;
			default:
				return 13;
		}
	}
	
	private boolean isInputReLU(Hop input) {
		return input instanceof UnaryOp && ((UnaryOp) input).getOp() == OpOp1.SELP;
	}
	
	private boolean canFuseIntoReluMaxPooling(ExecType et, ArrayList<Hop> inputs) {
		return op == ConvOp.MAX_POOLING && (et == ExecType.CP || et == ExecType.SPARK) && isInputReLU(inputs.get(0));
	}
	
	private static class CustomPair {
		public Hop hop1;
		public Hop hop2;
		public Hop hop3;
		public ConvolutionOp op1;
		public ConvolutionOp op2;
	}
	
	private Hop getInput(Hop input, int index) {
		return (input.getInput() != null && input.getInput().size() > index) ? input.getInput().get(index) : null;
	}
	
	private AggBinaryOp getFirstInputIfAggBinary(Hop input) {
		Hop firstInput = getInput(input, 0);
		return (firstInput != null && firstInput instanceof AggBinaryOp) ? (AggBinaryOp)firstInput : null;
	}
	
	private ConvolutionOp getInputIfConvolution(Hop input, int index) {
		Hop ret = getInput(input, index);
		return (ret != null && ret instanceof ConvolutionOp) ? (ConvolutionOp)ret : null;
	}
	
	// Returns null if pattern matches, else returns 1 and 2
	// 1 --- op1 --- %*% --- op3
	// 2 --- op2 -----|
	public boolean patternMatchHopTree(Hop op3, ConvOp op1Type, ConvOp op2Type, ConvOp op3Type, CustomPair ret) {
		AggBinaryOp aggBinOp = null;
		
		if(op3Type == ConvOp.PASS) {
			aggBinOp = op3 instanceof AggBinaryOp ? ((AggBinaryOp)op3) : null;
		}
		else if(op3 instanceof ConvolutionOp && ((ConvolutionOp)op3).getOp() == op3Type) {
			aggBinOp = getFirstInputIfAggBinary(op3);
		}
		
		Hop hop1 = null; ConvolutionOp op1 = null; ConvolutionOp op2 = null;
		if(aggBinOp != null) {
			if(op1Type == ConvOp.PASS) {
				hop1 = getInput(aggBinOp, 0);
			}
			else if(getInputIfConvolution(aggBinOp, 0).getOp() == op1Type) {
				op1 = getInputIfConvolution(aggBinOp, 0);
				hop1 = getInput(op1, 0);
			}
		}
		Hop hop2 = null;
		if(aggBinOp != null) {
			if(op2Type == ConvOp.PASS) {
				hop2 = getInput(aggBinOp, 1);
			}
			else if(getInputIfConvolution(aggBinOp, 1).getOp() == op2Type) {
				op2 = getInputIfConvolution(aggBinOp, 1);
				hop2 = getInput(op2, 0);
			}
		}
		ret.hop1 = hop1;
		ret.hop2 = hop2;
		ret.op1 = op1;
		ret.op2 = op2;
		return hop1 != null && hop2 != null ? true : false;
	}
	
	
	private boolean canFuseIntoConv2dBiasAdd(ExecType et, ArrayList<Hop> inputs, CustomPair ret) {
		if(op == ConvOp.BIAS_ADD && 
			(	et == ExecType.CP ||
				(et == ExecType.SPARK && getInput(this, 0).getColsInBlock() > 0 &&  getInput(this, 0)._dim2 <= MAPSIDE_THRESHOLD))) {
			// filter ---  PASS  --- %*% --- RESHAPE_COL --- BIAS_ADD
			// image  --- IM2COL -----|
			boolean matched = patternMatchHopTree(getInput(this, 0), ConvOp.PASS, ConvOp.IM2COL, ConvOp.RESHAPE_COL, ret);
			ret.hop3 = matched ? getInput(this, 1) : null;
			return matched;
		}
		return false;
	}
	
	private boolean canFuseIntoConv2d(ExecType et, ArrayList<Hop> inputs, CustomPair ret) {
		if(et == ExecType.CP || et == ExecType.GPU ||
			(et == ExecType.SPARK && getInput(this, 0).getColsInBlock() > 0 &&  getInput(this, 0)._dim2 <= MAPSIDE_THRESHOLD)) {
			// filter ---  PASS  --- %*% --- RESHAPE_COL
			// image  --- IM2COL -----|
			return patternMatchHopTree(this, ConvOp.PASS, ConvOp.IM2COL, ConvOp.RESHAPE_COL, ret);
		}
		return false;
	}
	
	private boolean canFuseIntoConv2dBackwardFilter(ExecType et, ArrayList<Hop> inputs, CustomPair ret) {
		if(et == ExecType.CP || et == ExecType.GPU ||
			(et == ExecType.SPARK && getInput(this, 0).getColsInBlock() > 0 &&  getInput(this, 0).getColsInBlock() <= MAPSIDE_THRESHOLD)) {
			// image ---   IM2COL  --- %*% --- Transpose --- PASS
			// dout  --- ROTATE180 -----|
			if(op == ConvOp.PASS && getInput(this, 0) instanceof ReorgOp) {
				return patternMatchHopTree(getInput(getInput(this, 0), 0), ConvOp.IM2COL, ConvOp.ROTATE180, ConvOp.PASS, ret);
			}
		}
		return false;
	}
	
	private boolean canFuseIntoConv2dBackwardData(ExecType et, ArrayList<Hop> inputs, CustomPair ret) {
		if(et == ExecType.CP || et == ExecType.GPU ||
			(et == ExecType.SPARK && getInput(this, 0).getColsInBlock() > 0 &&  getInput(this, 0).getColsInBlock() <= MAPSIDE_THRESHOLD)) {
			// dout  ---  ROTATE80  --- %*% --- COL2IM
			// filter  ---  PASS --------|
			return patternMatchHopTree(this, ConvOp.ROTATE180, ConvOp.PASS, ConvOp.COL2IM, ret);
		}
		return false;
	}
	
	private ConvolutionTransform createConvolutionTransformLop(Lop in, OperationTypes lopOp, ExecType et) throws HopsException {
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		ConvolutionTransform transform1 = new ConvolutionTransform(in, lopOp, getDataType(), getValueType(), et, k);
		setOutputDimensions(transform1);
		
		setLineNumbers(transform1);
		return transform1;
	}
	
	private ConvolutionTransform createConvolutionTransformLop(Lop in, Lop in2, OperationTypes lopOp, ExecType et) throws HopsException {
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		ConvolutionTransform transform1 = new ConvolutionTransform(in, lopOp, getDataType(), getValueType(), et, k);
		setOutputDimensions(transform1);
		setLineNumbers(transform1);
		transform1.addInput(in2);
		in2.addOutput(transform1);
		return transform1;
	}
	
	private ConvolutionTransform createConvolutionTransformLop(Lop in, Lop in2, Lop in3, OperationTypes lopOp, ExecType et) throws HopsException {
		int k = OptimizerUtils.getConstrainedNumThreads(_maxNumThreads);
		ConvolutionTransform transform1 = new ConvolutionTransform(in, lopOp, getDataType(), getValueType(), et, k);
		setOutputDimensions(transform1);
		setLineNumbers(transform1);
		transform1.addInput(in2);
		in2.addOutput(transform1);
		transform1.addInput(in3);
		in3.addOutput(transform1);
		return transform1;
	}
	
	private ConvolutionTransform addOutputLops(ConvolutionTransform transform1, ArrayList<Hop> inputs1, int startIndex) throws HopsException, LopsException {
		// stride1, stride2, padding1, padding2  
		// input_shape1, input_shape2, input_shape3, input_shape4, 
		// filter_shape1, filter_shape2, filter_shape3, filter_shape4
		for( int i=startIndex; i < inputs1.size(); i++ )
		{
			Lop ltmp = inputs1.get(i).constructLops();
			transform1.addInput(ltmp);
			ltmp.addOutput(transform1);
		}
		transform1.setLevel(); //force order of added lops
		return transform1;
	}
	
	public Lop constructConvolutionLops(ExecType et) throws HopsException, LopsException {
		ArrayList<Hop> inputs = getInput();
		if(inputs.size() != getNumExpectedInputs()) 
			throw new HopsException("Incorrect number of inputs for " + op.name());
		
		Lop in = null; 
		CustomPair tmp = new CustomPair();
		if(canFuseIntoReluMaxPooling(et, inputs)) {
			// Fuse RELU + MAX_POOLING into RELU_MAX_POOLING
			in = inputs.get(0).getInput().get(0).constructLops();
			return addOutputLops(createConvolutionTransformLop(in, OperationTypes.RELU_MAX_POOLING, et), inputs, 1);
		}
		else if(canFuseIntoConv2dBiasAdd(et, inputs, tmp)) {
			// Fuse following hop tree into DIRECT_CONV2D_BIAS_ADD
			// filter ---  PASS  --- %*% --- RESHAPE_COL --- BIAS_ADD
			// image  --- IM2COL -----|
			Lop image = tmp.hop2.constructLops();
			Lop filter = tmp.hop1.constructLops();
			Lop bias = tmp.hop3.constructLops();
			return addOutputLops(createConvolutionTransformLop(image, bias, filter, OperationTypes.DIRECT_CONV2D_BIAS_ADD, et), tmp.op2.getInput(), 1);
		}
		else if(canFuseIntoConv2d(et, inputs, tmp)) {
			// Fuse following hop tree into DIRECT_CONV2D
			// filter ---  PASS  --- %*% --- RESHAPE_COL
			// image  --- IM2COL -----|
			Lop image = tmp.hop2.constructLops();
			Lop filter = tmp.hop1.constructLops();
			return addOutputLops(createConvolutionTransformLop(image, filter, OperationTypes.DIRECT_CONV2D, et), tmp.op2.getInput(), 1);
		}
		else {
			in = inputs.get(0).constructLops();
			return addOutputLops(createConvolutionTransformLop(in, HopsConv2Lops.get(op), et), inputs, 1);
		}
	}

			
	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{		
		double sparsity = 1.0;
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
		// [numRows, numCols, NNZ] 
		long[] ret = null;
		
		if(op == ConvOp.BIAS_ADD) {
			MatrixCharacteristics[] mc = memo.getAllInputStats(getInput());
			ret = new long[3];
			ret[0] = mc[0].rowsKnown() ? mc[0].getRows() : -1;
			ret[1] = mc[0].colsKnown() ? mc[0].getCols() : -1;
			ret[2] = -1;
			return ret;
		}
	
		ConvolutionParameters params;
		try {
			params = parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		ret = new long[3];
		switch(op) 
		{
			case MAX_POOLING:
			{
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getExtractedVal(params.C, params.P, params.Q);
				ret[2] = -1;
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getInput().get(0)._dim2;
				ret[2] = -1;
				break;
			}
			case IM2COL:
			{
				ret[0] = getExtractedVal(params.C, params.R, params.S);
				ret[1] = getExtractedVal(params.N, params.P, params.Q);
				ret[2] = -1;
				break;
			}
			case RESHAPE_COL:
			{
				ret[0] = params.N;
				ret[1] = getExtractedVal(params.K, params.P, params.Q);
				ret[2] = getInput().get(0).getNnz();
				break;
			}
			case ROTATE180:
			{
				ret[0] = getExtractedVal(params.N, params.P, params.Q);
				ret[1] = params.K;
				ret[2] = getInput().get(0).getNnz();
				break;
			}
			case COL2IM:
			{
				ret[0] = params.N;
				ret[1] = getExtractedVal(params.C, params.H, params.W);
				ret[2] = -1;
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				ret[0] = getInput().get(1)._dim1;
				ret[1] = getInput().get(1)._dim2;
				ret[2] = -1;
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				ret[0] = getInput().get(0)._dim1;
				ret[1] = getInput().get(0)._dim2;
				ret[2] = -1;
				break;
			}
			default:
				throw new RuntimeException("Unsupported op:" + op.name());
		}
		
		if(LOG.isDebugEnabled() && (ret[0] <= 0 || ret[1] <= 0)) {
			LOG.debug("Unknown dimensions for ConvolutionOp in inferOutputCharacteristics:" + op.name() + " " + ret[0] + " " + ret[1] + 
					" img_dim=[" + params.N + " " + params.C + " " + params.H + " " + params.W + "]" +
					" filter_dim=[" + params.K + " " + params.C + " " + params.H + " " + params.W + "]" + 
					" output_feature_map=[" + params.P + " " + params.Q + "] stride=[" + params.stride_h + " " + params.stride_w + "]" +
					" pad=[" + params.pad_h + " " + params.pad_w + "]");
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
				_etype = findGPUExecTypeByMemEstimate(findExecTypeByMemEstimate());
				// TODO: Fix this after adding remaining spark instructions
				// _etype = !isEligibleForSpark() && _etype == REMOTE ?  ExecType.CP : _etype;
			}
			else {
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( ConfigurationManager.isDynamicRecompilation() && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
		
		return _etype;
	}
	
	// stride1, stride2, padding1, padding2  
	// input_shape1, input_shape2, input_shape3, input_shape4, 
	// filter_shape1, filter_shape2, filter_shape3, filter_shape4
	ConvolutionParameters parseInput() throws DMLRuntimeException {
		ConvolutionParameters params = null;
		if(op == ConvOp.MAX_POOLING_BACKWARD 
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_FILTER
				|| op == ConvOp.DIRECT_CONV2D_BACKWARD_DATA) {
			params = new ConvolutionParameters(
					computeSizeInformation(getInput().get(6)),
					computeSizeInformation(getInput().get(7)), 
					computeSizeInformation(getInput().get(8)), 
					computeSizeInformation(getInput().get(9)), 
					computeSizeInformation(getInput().get(10)), 
					computeSizeInformation(getInput().get(12)), 
					computeSizeInformation(getInput().get(13)), 
					computeSizeInformation(getInput().get(2)), 
					computeSizeInformation(getInput().get(3)), 
					computeSizeInformation(getInput().get(4)), 
					computeSizeInformation(getInput().get(5)), _maxNumThreads);
		}
		else {
			params = new ConvolutionParameters(
					computeSizeInformation(getInput().get(5)),
					computeSizeInformation(getInput().get(6)), 
					computeSizeInformation(getInput().get(7)), 
					computeSizeInformation(getInput().get(8)), 
					computeSizeInformation(getInput().get(9)), 
					computeSizeInformation(getInput().get(11)), 
					computeSizeInformation(getInput().get(12)), 
					computeSizeInformation(getInput().get(1)), 
					computeSizeInformation(getInput().get(2)), 
					computeSizeInformation(getInput().get(3)), 
					computeSizeInformation(getInput().get(4)), _maxNumThreads);
		}
		return params;
	}

	public static long getExtractedVal(long val1, long val2, long val3) {
		if(val1 == -1 || val2 == -1 || val3 == -1) {
			return -1;
		}
		return val1*val2*val3;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		if(op == ConvOp.BIAS_ADD) {
			Hop input1 = getInput().get(0);
			setDim1(input1.getDim1());
			setDim2(input1.getDim2());
			return;
		}
		
		ConvolutionParameters params;
		try {
			params = parseInput();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
		
		switch(op) 
		{
			case MAX_POOLING:
			{	
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getExtractedVal(params.C, params.P, params.Q);
				_nnz = -1; // cannot infer stats
				break;
			}
			case MAX_POOLING_BACKWARD:
			{
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getInput().get(0)._dim2;
				_nnz = -1;
				break;
			}
			case IM2COL:
			{
				_dim1 = getExtractedVal(params.C, params.R, params.S);
				_dim2 = getExtractedVal(params.N, params.P, params.Q);
				_nnz = -1;
				break;
			}
			case RESHAPE_COL:
			{
				_dim1 = params.N;
				_dim2 = getExtractedVal(params.K, params.P, params.Q);
				_nnz = getInput().get(0).getNnz();
				break;
			}
			case ROTATE180:
			{
				_dim1 = getExtractedVal(params.N, params.P, params.Q);
				_dim2 = params.K;
				_nnz = getInput().get(0).getNnz();
				break;
			}
			case COL2IM:
			{
				_dim1 = params.N;
				_dim2 = getExtractedVal(params.C, params.H, params.W);
				_nnz = -1;
				break;
			}
			case DIRECT_CONV2D_BACKWARD_DATA:
			{
				_dim1 = getInput().get(0)._dim1;
				_dim2 = getInput().get(0)._dim2;
				_nnz = -1; // cannot infer stats
				break;
			}
			case DIRECT_CONV2D_BACKWARD_FILTER:
			{
				_dim1 = getInput().get(1)._dim1;
				_dim2 = getInput().get(1)._dim2;
				_nnz = -1; // cannot infer stats
				break;
			}
			default:
				throw new RuntimeException("The sizes are not refreshed for " + op.name());
		}
		
		if(LOG.isDebugEnabled() && (_dim1 <= 0 || _dim2 <= 0)) {
			LOG.debug("Unknown dimensions for ConvolutionOp in refreshSizeInformation:" + op.name() + " " + _dim1 + " " + _dim2 + 
					" img_dim=[" + params.N + " " + params.C + " " + params.H + " " + params.W + "]" +
					" filter_dim=[" + params.K + " " + params.C + " " + params.H + " " + params.W + "]" + 
					" output_feature_map=[" + params.P + " " + params.Q + "] stride=[" + params.stride_h + " " + params.stride_w + "]" +
					" pad=[" + params.pad_h + " " + params.pad_w + "]");
		}
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		ConvolutionOp ret = new ConvolutionOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret.op = op;
		ret._maxNumThreads = _maxNumThreads;
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof ConvolutionOp) )
			return false;
		
		ConvolutionOp that2 = (ConvolutionOp)that;
		
		boolean ret =  (op == that2.op)
				    && (getInput().size()==that.getInput().size())
				    && _maxNumThreads == that2._maxNumThreads;
		
		//compare all childs
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

	@Override
	public void setMaxNumThreads( int k ) {
		_maxNumThreads = k;
	}
	
	@Override
	public int getMaxNumThreads() {
		return _maxNumThreads;
	}
}

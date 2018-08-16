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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOpDnn;
import static org.apache.sysml.hops.rewrite.HopDagPatternMatcher.*;

/*
 * This class contains GPU-specific rewrites.
 */
public class RewriteGPUSpecificOps extends HopRewriteRule {
	
	// -------------------------------------------------------------------------------------------
	// Pattern 1:
	// subgrp_vars = matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
	// var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
	private static final HopDagPatternMatcher _batchNormUpdatedVar; 
	static {
		HopDagPatternMatcher subgrp_vars = matrix( 
				mult(colVars(leaf("X")), leaf("varConst1").isScalar()), 
				leaf("C").isScalar(),  
				leaf("HW").isScalar());
		_batchNormUpdatedVar = mm_plus( rowMeans(subgrp_vars), mult(rowVars(leaf("subgrp_means")),  leaf("varConst2").isScalar()));
	}
		
	// Pattern 2:
	private static final HopDagPatternMatcher _batchNormTest;
	static {
		// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
		HopDagPatternMatcher norm = 
				bias_multiply(
						bias_add(leaf("X"), unaryMinus(leaf("mean"))), 
						div(1, sqrt(plus(leaf("var"), leaf("eps")))));
		// hi = bias_add(bias_multiply(norm, gamma), beta)
		_batchNormTest = bias_add(
				bias_multiply(norm, leaf("gamma")), 
				leaf("beta"))
			.fitsOnGPU(3); // 2x for input and output and 1x for overhead
	}
	
	// Pattern 3:
	// rowSums(matrix(colSums(X), rows=C, cols=HW))
	private static final HopDagPatternMatcher _channelSums = 
		rowSums(matrix(	colSums(
						leaf("X").fitsOnGPU(2)), 
						leaf("C").isScalar(), 
						leaf("HW").isScalar()));
	
	// Pattern 4:
	// X - mu*v_prev + (1+mu)*v
	private static final HopDagPatternMatcher _updateNesterovX = 
		mm_plus(	minus(leaf("X"), mult(leaf("mu").isScalar(), leaf("v_prev"))), 	// X - mu*v_prev
			mult(leaf("onePlusMu").isScalar(), leaf("v")))						// (1+mu)*v
		.fitsOnGPU(3); // 2x for input and output and 1x for overhead;
	
	// Pattern 5:
	// matrix(colMeans(X), rows=C, cols=Hin*Win)
	// This avoids unnecessary copy by the reshape operator
	private static final HopDagPatternMatcher _reshapeColMeans = 
			matrix(colMeans(leaf("X").fitsOnGPU(2)), leaf("C").isScalar(), leaf("HW").isScalar());
	
	// -------------------------------------------------------------------------------------------
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( int i = 0; i < roots.size(); i++ )
			rule_GPUKernels(roots, roots.get(i), false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for( int i = 0; i < roots.size(); i++ )
			rule_GPUKernels(roots, roots.get(i), true );
		Hop.resetVisitStatus(roots, true);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		rule_GPUKernels(null, root, false );
		
		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		rule_GPUKernels(null, root, true );
		
		return root;
	}
	
	/**
	 * Fuse the kernel
	 * 
	 * @param roots root operators
	 * @param hop high-level operator
	 * @param descendFirst true if recursively process children first
	 */
	private void rule_GPUKernels(ArrayList<Hop> roots, Hop hop, boolean descendFirst) 
	{
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++) {
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_GPUKernels(roots, hi, descendFirst); //see below
			
			hi = batchNormUpdatedVars(hop, hi, i);
			hi = batchNormTest(hop, hi, i); 
			hi = channelSums(hop, hi, i); 
			hi = updateNesterovX(hop, hi, i);
			hi = reshapeColMeans(hop, hi, i);
	
			if( !descendFirst )
				rule_GPUKernels(roots, hi, descendFirst);
		}

		hop.setVisited();
	}
	
	/**
	 * Checks for the channelSums pattern (output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize)))
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop channelSums(Hop parent, Hop hi, int pos) {
		if(_channelSums.matches(hi)) {
			LOG.debug("Applied channelSums rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_channelSums, OpOpDnn.CHANNEL_SUMS, "X", "C", "HW");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	}
	
	/**
	 * Checks for the nesterov_update_x pattern (X = X - mu*v_prev + (1+mu)*v)
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop updateNesterovX(Hop parent, Hop hi, int pos) {
		if(_updateNesterovX.matches(hi) && 
				(1+_updateNesterovX.getLiteralValue("mu")) == _updateNesterovX.getLiteralValue("onePlusMu")) {
			Hop X = _updateNesterovX.getMatchedHop("X");
			Hop v = _updateNesterovX.getMatchedHop("v");
			Hop v_prev = _updateNesterovX.getMatchedHop("v_prev");
			if(hasSameDimensions(X, v) && hasSameDimensions(X, v_prev)) {
				LOG.debug("Applied updateNesterovX rewrite.");
				Hop newHop = HopRewriteUtils.createDnnOp(_updateNesterovX, OpOpDnn.UPDATE_NESTEROV_X, "X", "v", "v_prev", "mu");
				return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
			}
		}
		return hi;
	}
	
	/**
	 * Checks for the reshapeColMeans pattern (matrix(colMeans(X), rows=C, cols=Hin*Win))
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop reshapeColMeans(Hop parent, Hop hi, int pos) {
		if(_reshapeColMeans.matches(hi)) {
			LOG.debug("Applied reshapeColMeans rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_reshapeColMeans, OpOpDnn.RESHAPE_COLMEANS, "X", "C", "HW");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	}
	
	
	
	
	private static boolean hasSameDimensions(Hop x, Hop y) {
		return x.dimsKnown() && y.dimsKnown() && (x.getDim1() == y.getDim1()) && (x.getDim2() == y.getDim2());
	}
	
	/**
	 * Checks for the updated variance pattern for batch norm
	 * and returns a new DnnOp if matched
	 * 
	 * subgrp_vars = matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
	 * var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop batchNormUpdatedVars(Hop parent, Hop hi, int pos) {
		if(_batchNormUpdatedVar.matches(hi)) {
			double HW = _batchNormUpdatedVar.getLiteralValue("HW");
			if(_batchNormUpdatedVar.getLiteralValue("varConst2") == ((HW-1)/HW)) {
				LOG.debug("Applied batchNormUpdatedVar rewrite.");
				Hop newHop = HopRewriteUtils.createDnnOp(_batchNormUpdatedVar, OpOpDnn.UPDATE_EMA_VAR, 
						// varConst1 => ((N-1)/N)
						"subgrp_means", "X", "C", "HW", "varConst1");
				return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
			}
		}
		return hi;
	}
	
	/**
	 * Checks for the batch norm (mode="test") pattern using the helper isBatchNormTrainMean and isBatchNormTrainVar
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop batchNormTest(Hop parent, Hop hi, int pos) {
		if(_batchNormTest.matches(hi)) {
			LOG.debug("Applied batchNormTest rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_batchNormTest, OpOpDnn.BATCH_NORM2D_TEST, "X", "gamma", "beta", "mean", "var", "eps");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	}
}

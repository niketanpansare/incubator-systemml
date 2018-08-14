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
import java.util.HashMap;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.hops.DnnOp;
import static org.apache.sysml.hops.rewrite.HopDagPatternMatcher.*;

/*
 * This class contains GPU-specific rewrites for following patterns:
 * 
 * 1. batchNormTest: applied when mode="test" in batch normalization nn layer.
 * norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
 * hi = bias_add(bias_multiply(norm, gamma), beta)
 * 
 * 2. channelSums:
 * output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
 * 
 * 3. batchNormUpdatedMean:
 * ema_mean_upd = mu*ema_mean + (1-mu)*rowMeans(subgrp_means)
 * 
 * 4. updateNesterovX:
 * X = X - mu*v_prev + (1+mu)*v
 * 
 */
public class RewriteGPUSpecificOps extends HopRewriteRule {
	

	private static HopDagPatternMatcher _batchNormUpdatedMean = null;
	private static HopDagPatternMatcher _batchNormTest = null;
	private static HopDagPatternMatcher _channelSums = null;
	private static HopDagPatternMatcher _updateNesterovX = null;
	
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
			
			hi = batchNormUpdatedMean(hop, hi, i);
			hi = batchNormTest(hop, hi, i); 
			hi = channelSums(hop, hi, i); 
			hi = updateNesterovX(hop, hi, i);
	
			if( !descendFirst )
				rule_GPUKernels(roots, hi, descendFirst);
		}

		hop.setVisited();
	}
	
	private static Hop channelSums(Hop parent, Hop hi, int pos) {
		if(_channelSums == null) {
			// 1. Create placeholders for the input variables
			HashMap<String, HopDagPatternMatcher> in = createPlaceholders("X", "C", "HW");
			in.get("X").addPredicate("fitsOnGPU", HopDagPatternMatcher.fitsOnGPU(2));
			in.get("C").isScalar();
			in.get("HW").isScalar();
			
			// 2. Create the pattern matcher
			// output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
			_channelSums = rowSums(matrix(colSums(in.get("X")), in.get("C"), in.get("HW")));
			
			// 3. Register the placeholders to fetch the Hop if the pattern matches
			_channelSums.register(in); 
		}
		
		if(_channelSums.matches(hi)) {
			ArrayList<Hop> inHops = new ArrayList<Hop>();
			inHops.add(_channelSums.getHop("X"));
			inHops.add(_channelSums.getHop("C"));
			inHops.add(_channelSums.getHop("HW"));
			LOG.debug("Applied channelSums rewrite.");
			Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
					OpOpDnn.CHANNEL_SUMS, inHops);
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
		if(_updateNesterovX == null) {
			// 1. Create placeholders for the input variables
			HashMap<String, HopDagPatternMatcher> in = createPlaceholders("X", "v", "v_prev", "mu", "onePlusMu");
			in.get("mu").isScalar();
			in.get("onePlusMu").isScalar();
			
			// 2. Create the pattern matcher
			// X = X - mu*v_prev + (1+mu)*v
			_updateNesterovX = 
					plus(
							minus(in.get("X"), mult(in.get("mu"), in.get("v_prev"))), 	// X - mu*v_prev
							mult(in.get("onePlusMu"), in.get("v")))						// (1+mu)*v
					.addPredicate("fitsOnGPU", HopDagPatternMatcher.fitsOnGPU(3)); // 2x for input and output and 1x for overhead
					
			
			// 3. Register the placeholders to fetch the Hop if the pattern matches
			_updateNesterovX.register(in); 
		}
		
		if(_updateNesterovX.matches(hi) && 
				(1+_updateNesterovX.getLiteralValue("mu")) == _updateNesterovX.getLiteralValue("onePlusMu")) {
			Hop X = _updateNesterovX.getHop("X");
			Hop v = _updateNesterovX.getHop("v");
			Hop v_prev = _updateNesterovX.getHop("v_prev");
			if(hasSameDimensions(X, v) && hasSameDimensions(X, v_prev)) {
				ArrayList<Hop> inHops = new ArrayList<Hop>();
				inHops.add(X);
				inHops.add(v);
				inHops.add(v_prev);
				inHops.add(_updateNesterovX.getHop("mu"));
				LOG.debug("Applied updateNesterovX rewrite.");
				Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
						OpOpDnn.UPDATE_NESTEROV_X, inHops);
				return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
			}
		}
		return hi;
	}
	
	private static boolean hasSameDimensions(Hop x, Hop y) {
		return x.dimsKnown() && y.dimsKnown() && (x.getDim1() == y.getDim1()) && (x.getDim2() == y.getDim2());
	}
	
	/**
	 * Checks for the updated mean pattern for batch norm
	 * and returns a new DnnOp if matched
	 * 
	 * mean = rowMeans(subgrp_means)
	 * ema_mean_upd = mu*ema_mean + (1-mu)*mean
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop batchNormUpdatedMean(Hop parent, Hop hi, int pos) {
		if(_batchNormUpdatedMean == null) {
			// 1. Create placeholders for the input variables
			HashMap<String, HopDagPatternMatcher> in = createPlaceholders("ema_mean", "subgrp_means", "mu", "oneMinusMu");
			in.get("subgrp_means").addPredicate("fitsOnGPU", HopDagPatternMatcher.fitsOnGPU(2));
			in.get("mu").isScalar();
			in.get("oneMinusMu").isScalar();
			
			// 2. Create the pattern matcher
			// ema_mean_upd = mu*ema_mean + (1-mu)*rowMeans(subgrp_means)
			_batchNormUpdatedMean = 
					plus(	mult(in.get("mu"), in.get("ema_mean")),  
							mult(in.get("oneMinusMu"), rowMeans(in.get("subgrp_means"))));
			
			// 3. Register the placeholders to fetch the Hop if the pattern matches
			_batchNormUpdatedMean.register(in); 
		}
		
		if(_batchNormUpdatedMean.matches(hi) && 
				(1-_batchNormUpdatedMean.getLiteralValue("mu")) == _batchNormUpdatedMean.getLiteralValue("oneMinusMu")) {
			LOG.debug("Applied batchNormUpdatedMean rewrite.");
			ArrayList<Hop> inHops = new ArrayList<Hop>();
			inHops.add(_batchNormUpdatedMean.getHop("ema_mean"));
			inHops.add(_batchNormUpdatedMean.getHop("subgrp_means"));
			inHops.add(_batchNormUpdatedMean.getHop("mu"));
			Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
					OpOpDnn.UPDATE_EMA_MEAN, inHops);
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
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
		if(_batchNormTest == null) {
			// 1. Create placeholders for the input variables
			HashMap<String, HopDagPatternMatcher> in = createPlaceholders("X", "mean", "var", "eps", "gamma", "beta");
			in.get("eps").isScalar();
			
			// 2. Create the pattern matcher
			// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
			HopDagPatternMatcher norm = 
					bias_multiply(
							bias_add(in.get("X"), unaryMinus(in.get("mean"))), 
							div(1, sqrt(plus(in.get("var"), in.get("eps")))));
			// hi = bias_add(bias_multiply(norm, gamma), beta)
			_batchNormTest = bias_add(
					bias_multiply(norm, in.get("gamma")), 
					in.get("beta"))
				.addPredicate("fitsOnGPU", HopDagPatternMatcher.fitsOnGPU(3)); // 2x for input and output and 1x for overhead
			
			// 3. Register the placeholders to fetch the Hop if the pattern matches
			_batchNormTest.register(in); 
			
		}
		
		if(_batchNormTest.matches(hi)) {
			ArrayList<Hop> inHops = new ArrayList<Hop>();
			inHops.add(_batchNormTest.getHop("X"));
			inHops.add(_batchNormTest.getHop("gamma"));
			inHops.add(_batchNormTest.getHop("beta"));
			inHops.add(_batchNormTest.getHop("mean"));
			inHops.add(_batchNormTest.getHop("var"));
			inHops.add(_batchNormTest.getHop("eps"));
			LOG.debug("Applied batchNormTest rewrite.");
			Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
					OpOpDnn.BATCH_NORM2D_TEST, inHops);
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	}
}

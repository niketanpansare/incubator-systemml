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
import java.util.List;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.function.Predicate;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.utils.Explain;

public class HopDagPatternMatcher {
	List<HopPredicate> _predicates = new ArrayList<>();
	List<HopDagPatternMatcher> _children = new ArrayList<>();
	Hop current = null;
	HashMap<String, HopDagPatternMatcher> _cache = null;
	private static final boolean DEBUG_REWRITES = false;
	
	// Simple utility for debugging the rewrites
	public static class HopPredicate implements Predicate<Hop> {
		final String _name;
		final Function<Hop, Boolean> _pred;
		public HopPredicate(String name, Function<Hop, Boolean> pred) {
			_name = name;
			_pred = pred;
		}
		@Override
		public boolean test(Hop h) {
			return _pred.apply(h);
		}
		@Override
	    public String toString() {
	        return _name;
	    }
	}
	
	public static HashMap<String, HopDagPatternMatcher> createPlaceholders(String... varNames) {
		HashMap<String, HopDagPatternMatcher> ret = new HashMap<>();
		for(String varName : varNames) {
			ret.put(varName, new HopDagPatternMatcher());
		}
		return ret;
	}
	
	/**
	 * Adds a predicate to the pattern matcher
	 * 
	 * @param name name of the pattern for debugging
	 * @param pred higher order function that takes as an input a hop and returns true if the pattern matches else false
	 * @return this
	 */
	public HopDagPatternMatcher addPredicate(String name, Function<Hop, Boolean> pred) {
		_predicates.add(new HopPredicate(name, pred));
		return this;
	}
	public HopDagPatternMatcher addChild(HopDagPatternMatcher... children) {
		for(int i = 0; i < children.length; i++) {
			_children.add(children[i]);
		}
		return this;
	}
	public HopDagPatternMatcher register(HashMap<String, HopDagPatternMatcher> placeholders) {
		_cache = placeholders;
		return this;
	}
	public Hop getHop(String varName) {
		if(_cache == null || !_cache.containsKey(varName)) {
			throw new RuntimeException("Incorrect usage: the variable " + varName + " is not registered as input.");
		}
		HopDagPatternMatcher val = _cache.get(varName);
		if(val.current == null) {
			throw new RuntimeException("getHop should be called only if evaluate succeeds.");
		}
		return val.current;
	}
	public double getLiteralValue(String varName) {
		return OptimizerUtils.rEvalSimpleDoubleExpression(getHop(varName), new HashMap<>());
	}
	@Override
    public String toString() {
        return ( _predicates.size() == 1 ? _predicates.get(0).toString() : "" ) + ( current == null ? "" : " for hop:" +  Explain.explain(current));
    }
	public boolean matches(Hop h) {
		current = h;
		if(h == null || (_children.size() > 0 && h.getInput().size() < _children.size())) {
			if(DEBUG_REWRITES) {
				System.out.println("The expected number of children (" + _children.size() + ") didnot match the number of inputs (" + h.getInput().size() + ") " + this);
			}
			return false;
		}
		for(HopPredicate p : _predicates) {
			if(!p.test(h)) {
				if(DEBUG_REWRITES) {
					System.out.println("The predicate " + p.toString() + " failed for " + Explain.explain(h));
				}
				return false;
			}
		}
		int index = 0;
		for(HopDagPatternMatcher child : _children) {
			if(!child.matches(h.getInput().get(index))) {
				return false;
			}
			index++;
		}
		return true;
	}
	public HopDagPatternMatcher isMatrix() {
		return addPredicate("isMatrix", h -> h.getDataType() == DataType.MATRIX);
	}
	public HopDagPatternMatcher isScalar() {
		return addPredicate("isScalar", h -> h.getDataType() == DataType.SCALAR);
	}
	
	// Factory methods:
	public static HopDagPatternMatcher dummy = new HopDagPatternMatcher();
	public static HopDagPatternMatcher empty() {
		return new HopDagPatternMatcher();
	}
	public static HopDagPatternMatcher rowMeans(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("rowMeans", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChild(child1);
	}
	public static HopDagPatternMatcher rowSums(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("rowSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChild(child1);
	}
	public static HopDagPatternMatcher colSums(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("colSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Col)
			.addChild(child1);
	}
	public static HopDagPatternMatcher matrix(HopDagPatternMatcher X, HopDagPatternMatcher rows, HopDagPatternMatcher cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_reshape", h -> HopRewriteUtils.isReorg(h, ReOrgOp.RESHAPE))
				.addChild(X, rows, cols);
	}
	
	public static HopDagPatternMatcher bias_add(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("bias_add", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher bias_multiply(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("bias_multiply", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher unaryMinus(HopDagPatternMatcher child) {
		return new HopDagPatternMatcher().addPredicate("unaryMinus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS)
				&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0))
				.addChild(dummy, child);
	}
	public static HopDagPatternMatcher sqrt(HopDagPatternMatcher child) {
		return new HopDagPatternMatcher().addPredicate("sqrt", h -> HopRewriteUtils.isUnary(h, OpOp1.SQRT))
				.addChild(child);
	}
	public static Function<Hop, Boolean> fitsOnGPU(double constant) {
		return h -> _fitsOnGPU(h, constant);
	}
	public static HopDagPatternMatcher div(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher div(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(dummy, child2);
	}
	public static HopDagPatternMatcher div(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1, dummy);
	}
	
	public static HopDagPatternMatcher plus(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher plus(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(dummy, child2);
	}
	public static HopDagPatternMatcher plus(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1, dummy);
	}
	
	public static HopDagPatternMatcher minus(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher minus(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(dummy, child2);
	}
	public static HopDagPatternMatcher minus(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1, dummy);
	}
	
	public static HopDagPatternMatcher mult(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT))
				.addChild(child1, child2);
	}
	public static HopDagPatternMatcher mult(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(dummy, child2);
	}
	public static HopDagPatternMatcher mult(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1, dummy);
	}
	
	private static boolean _fitsOnGPU(Hop h, double multiplier) {
		double memEst = multiplier*h.getMemEstimate();
		return DMLScript.USE_ACCELERATOR && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
}

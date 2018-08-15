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

import static org.apache.sysml.hops.rewrite.SimpleHopDagPatternMatcher.isScalar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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

public class SimpleHopDagPatternMatcher {
	private static final boolean DEBUG_REWRITES = false;
	
	// Predicates for the current HOP
	List<HopPredicate> _predicates = new ArrayList<>();
	// Child matchers
	List<SimpleHopDagPatternMatcher> _children = new ArrayList<>();
	private boolean isLeaf = false;
	
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
	
	/**
	 * Adds a predicate to the pattern matcher
	 * 
	 * @param name name of the pattern for debugging
	 * @param pred higher order function that takes as an input a hop and returns true if the pattern matches else false
	 * @return this
	 */
	public SimpleHopDagPatternMatcher addPredicate(String name, Function<Hop, Boolean> pred) {
		_predicates.add(new HopPredicate(name, pred));
		return this;
	}
	
	/**
	 * Add child pattern matcher
	 * @param children list of childer
	 * @return this
	 */
	public SimpleHopDagPatternMatcher addChildMatcher(SimpleHopDagPatternMatcher... children) {
		for(int i = 0; i < children.length; i++) {
			_children.add(children[i]);
		}
		return this;
	}
	
	/**
	 * Get the matched HOP DAGs
	 * @param varName variable names
	 * @return matched HOP
	 */
	public Hop getMatchedHop(String varName) {
		
		if(matchedHops == null || !matchedHops.containsKey(varName)) {
			throw new RuntimeException("Incorrect usage: the variable " + varName + " is not registered as input.");
		}
		return matchedHops.get(varName);
	}
	
	/**
	 * Return the value 
	 * 
	 * @param varName variable name
	 * @return the value of the LiteralOp 
	 */
	public double getLiteralValue(String varName) {
		return OptimizerUtils.rEvalSimpleDoubleExpression(getMatchedHop(varName), new HashMap<>());
	}
	
	@Override
    public String toString() {
        return _predicates.size() >= 1 ? _predicates.get(0).toString() : "";
    }
	
	/**
	 * Match the given HOP DAG
	 * 
	 * @param h root node of the HOP DAG 
	 * @return true if HOP DAG matches
	 */
	public boolean matches(Hop h) {
		return matchHelper(this, h);
	}
	
	private HashMap<String, Hop> matchedHops;
	private String variableName;
	private boolean matchHelper(SimpleHopDagPatternMatcher root, Hop h) {
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
		for(SimpleHopDagPatternMatcher child : _children) {
			if(!child.matchHelper(root, h.getInput().get(index))) {
				return false;
			}
			index++;
		}
		if(isLeaf) {
			if(root.matchedHops == null) {
				root.matchedHops = new HashMap<>();
			}
			root.matchedHops.put(variableName, h);
		}
		return true;
	}
	

	// Simple helper utilities for adding predicates
	public static Function<Hop, Boolean> fitsOnGPU(double constant) {
		return h -> _fitsOnGPU(h, constant);
	}
	
	// Factory methods:
	public static SimpleHopDagPatternMatcher dummy = new SimpleHopDagPatternMatcher();
	public static SimpleHopDagPatternMatcher rowMeans(SimpleHopDagPatternMatcher child1) {
		return new SimpleHopDagPatternMatcher().addPredicate("rowMeans", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChildMatcher(child1);
	}
	public SimpleHopDagPatternMatcher isScalar() {
		return this.addPredicate("isScalar", h -> h.getDataType() == DataType.SCALAR);
	}
	public static SimpleHopDagPatternMatcher leaf(String _variableName) {
		SimpleHopDagPatternMatcher ret = new SimpleHopDagPatternMatcher();
		ret.isLeaf = true;
		ret.variableName = _variableName;
		return ret;
	}
	public static SimpleHopDagPatternMatcher rowSums(SimpleHopDagPatternMatcher child1) {
		return new SimpleHopDagPatternMatcher().addPredicate("rowSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChildMatcher(child1);
	}
	public static SimpleHopDagPatternMatcher colSums(SimpleHopDagPatternMatcher child1) {
		return new SimpleHopDagPatternMatcher().addPredicate("colSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Col)
			.addChildMatcher(child1);
	}
	public static SimpleHopDagPatternMatcher matrix(SimpleHopDagPatternMatcher X, SimpleHopDagPatternMatcher rows, SimpleHopDagPatternMatcher cols) {
		return new SimpleHopDagPatternMatcher().addPredicate("matrix_reshape", h -> HopRewriteUtils.isReorg(h, ReOrgOp.RESHAPE))
				.addChildMatcher(X, rows, cols);
	}
	
	public static SimpleHopDagPatternMatcher bias_add(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("bias_add", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher bias_multiply(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("bias_multiply", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher unaryMinus(SimpleHopDagPatternMatcher child) {
		return new SimpleHopDagPatternMatcher().addPredicate("unaryMinus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS)
				&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0))
				.addChildMatcher(dummy, child);
	}
	public static SimpleHopDagPatternMatcher sqrt(SimpleHopDagPatternMatcher child) {
		return new SimpleHopDagPatternMatcher().addPredicate("sqrt", h -> HopRewriteUtils.isUnary(h, OpOp1.SQRT))
				.addChildMatcher(child);
	}
	public static SimpleHopDagPatternMatcher div(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher div(double child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static SimpleHopDagPatternMatcher div(SimpleHopDagPatternMatcher child1, double child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	public static SimpleHopDagPatternMatcher plus(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher plus(double child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static SimpleHopDagPatternMatcher plus(SimpleHopDagPatternMatcher child1, double child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	public static SimpleHopDagPatternMatcher minus(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher minus(double child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static SimpleHopDagPatternMatcher minus(SimpleHopDagPatternMatcher child1, double child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	public static SimpleHopDagPatternMatcher mult(SimpleHopDagPatternMatcher child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT))
				.addChildMatcher(child1, child2);
	}
	public static SimpleHopDagPatternMatcher mult(double child1, SimpleHopDagPatternMatcher child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static SimpleHopDagPatternMatcher mult(SimpleHopDagPatternMatcher child1, double child2) {
		return new SimpleHopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	private static boolean _fitsOnGPU(Hop h, double multiplier) {
		double memEst = multiplier*h.getMemEstimate();
		return DMLScript.USE_ACCELERATOR && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
}

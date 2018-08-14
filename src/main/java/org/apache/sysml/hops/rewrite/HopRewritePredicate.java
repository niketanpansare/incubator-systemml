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
import java.util.function.Function;
import java.util.function.Predicate;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.utils.Explain;

public class HopRewritePredicate {
	List<HopPredicate> _predicates = new ArrayList<>();
	List<HopRewritePredicate> _children = new ArrayList<>();
	Hop current = null;
	HashMap<String, HopRewritePredicate> _cache = null;
	private static final boolean DEBUG_REWRITES = false;
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
	public HopRewritePredicate addFilter(String name, Function<Hop, Boolean> pred) {
		_predicates.add(new HopPredicate(name, pred));
		return this;
	}
	public HopRewritePredicate addChild(HopRewritePredicate... children) {
		for(int i = 0; i < children.length; i++) {
			_children.add(children[i]);
		}
		return this;
	}
	public HopRewritePredicate registerHop(String varName, HopRewritePredicate pred) {
		if(_cache == null) {
			_cache = new HashMap<>();
		}
		_cache.put(varName, pred);
		return this;
	}
	public Hop getHop(String varName) {
		if(_cache == null || !_cache.containsKey(varName)) {
			throw new RuntimeException("Incorrect usage: the variable " + varName + " is not registered as input.");
		}
		HopRewritePredicate val = _cache.get(varName);
		if(val.current == null) {
			throw new RuntimeException("getHop should be called only if evaluate succeeds.");
		}
		return val.current;
	}
	public boolean matches(Hop h) {
		current = h;
		if(h == null || (_children.size() > 0 && h.getInput().size() < _children.size())) {
			if(DEBUG_REWRITES) {
				String suffix = ( _predicates.size() == 1 ? _predicates.get(0) : "" ) + " for hop:" + Explain.explain(h);
				System.out.println("The expected number of children (" + _children.size() + ") didnot match the number of inputs (" + h.getInput().size() + ") " + suffix);
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
		for(HopRewritePredicate child : _children) {
			if(!child.matches(h.getInput().get(index))) {
				return false;
			}
			index++;
		}
		return true;
	}
	public HopRewritePredicate isMatrix() {
		return addFilter("isMatrix", h -> h.getDataType() == DataType.MATRIX);
	}
	public HopRewritePredicate isScalar() {
		return addFilter("isScalar", h -> h.getDataType() == DataType.SCALAR);
	}
	
	// Factory methods:
	public static HopRewritePredicate empty() {
		return new HopRewritePredicate();
	}
	public static HopRewritePredicate bias_add(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("bias_add", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate bias_multiply(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("bias_multiply", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate unaryMinus(HopRewritePredicate child) {
		return new HopRewritePredicate().addFilter("unaryMinus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS)
				&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0))
				.addChild(child);
	}
	public static HopRewritePredicate sqrt(HopRewritePredicate child) {
		return new HopRewritePredicate().addFilter("sqrt", h -> HopRewriteUtils.isUnary(h, OpOp1.SQRT))
				.addChild(child);
	}
	public static Function<Hop, Boolean> fitsOnGPU(double constant) {
		return h -> _fitsOnGPU(h, constant);
	}
	public static HopRewritePredicate div(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate div(double child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(child2);
	}
	public static HopRewritePredicate div(HopRewritePredicate child1, double child2) {
		return new HopRewritePredicate().addFilter("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1);
	}
	
	public static HopRewritePredicate plus(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate plus(double child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(child2);
	}
	public static HopRewritePredicate plus(HopRewritePredicate child1, double child2) {
		return new HopRewritePredicate().addFilter("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1);
	}
	
	public static HopRewritePredicate minus(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate minus(double child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(child2);
	}
	public static HopRewritePredicate minus(HopRewritePredicate child1, double child2) {
		return new HopRewritePredicate().addFilter("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1);
	}
	
	public static HopRewritePredicate mult(HopRewritePredicate child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT))
				.addChild(child1, child2);
	}
	public static HopRewritePredicate mult(double child1, HopRewritePredicate child2) {
		return new HopRewritePredicate().addFilter("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChild(child2);
	}
	public static HopRewritePredicate mult(HopRewritePredicate child1, double child2) {
		return new HopRewritePredicate().addFilter("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChild(child1);
	}
	
	private static boolean _fitsOnGPU(Hop h, double multiplier) {
		double memEst = multiplier*h.getMemEstimate();
		return DMLScript.USE_ACCELERATOR && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
}

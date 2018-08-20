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
package org.apache.sysml.hops.rewrite

import java.util.{ArrayList, HashMap, List}
import java.util.function.{Function, Predicate}
import org.apache.sysml.api.DMLScript
import org.apache.sysml.hops._
import org.apache.sysml.hops.Hop._
import org.apache.sysml.parser.Expression.DataType
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool
import org.apache.sysml.utils.Explain
import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory
import org.apache.sysml.runtime.DMLRuntimeException

// See RewriteGPUSpecificOps for usage and design documentation
class HopPredicate(val _name:String, val _pred:Hop => Boolean) {
  override def toString() = _name
}

object HopDagPatternMatcher {
  val DEBUG_REWRITES = false;
  val LOG = LogFactory.getLog(classOf[HopDagPatternMatcher].getName())
  
  // Factory methods:
  val dummy = new HopDagPatternMatcher()
  def rowSums(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("rowSums", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.SUM, Direction.Row)).addChildMatcher(child)
  def rowMeans(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("rowMeans", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.MEAN, Direction.Row)).addChildMatcher(child)
	def rowVars(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("rowVars", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.VAR, Direction.Row)).addChildMatcher(child)
  def colSums(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("colSums", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.SUM, Direction.Col)).addChildMatcher(child)
	def colMeans(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("colMeans", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.MEAN, Direction.Col)).addChildMatcher(child)
	def colVars(child:HopDagPatternMatcher) = new HopDagPatternMatcher().addPredicate("colVars", 
        h => h.isInstanceOf[AggUnaryOp] && HopRewriteUtils.isAggUnaryOp(h.asInstanceOf[AggUnaryOp], AggOp.VAR, Direction.Col)).addChildMatcher(child)
  def leaf(variableName:String, dt:DataType=DataType.UNKNOWN):HopDagPatternMatcher = {
    val ret = new HopDagPatternMatcher(true)
    ret.variableName = variableName
    dt match {
      case DataType.SCALAR => ret.isScalar
      case DataType.MATRIX => ret.isMatrix
      case DataType.UNKNOWN => ret
      case _ => throw new DMLRuntimeException("Unsupported datatype in HopDagPatternMatcher for leaf")
    }
  }
  def matrix(X:HopDagPatternMatcher, rows:HopDagPatternMatcher, cols:HopDagPatternMatcher):HopDagPatternMatcher =
		new HopDagPatternMatcher().addPredicate("matrix_reshape", h => HopRewriteUtils.isReorg(h, ReOrgOp.RESHAPE))
		  .addChildMatcher(X, rows, cols)
	def bias_add(child1:HopDagPatternMatcher, child2:HopDagPatternMatcher):HopDagPatternMatcher =
		new HopDagPatternMatcher().addPredicate("bias_add", h => HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD))
		  .addChildMatcher(child1, child2)
	def bias_multiply(child1:HopDagPatternMatcher, child2:HopDagPatternMatcher):HopDagPatternMatcher =
		new HopDagPatternMatcher().addPredicate("bias_multiply", h => HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT))
		  .addChildMatcher(child1, child2)
	def sqrt(child:HopDagPatternMatcher):HopDagPatternMatcher =
		new HopDagPatternMatcher().addPredicate("sqrt", h => HopRewriteUtils.isUnary(h, OpOp1.SQRT))
		  .addChildMatcher(child)
	def fromDouble(d: Double) = new HopDagPatternMatcher()
    .addPredicate("fromDouble", h => HopRewriteUtils.isLiteralOfValue(h, d))
		.addChildMatcher(HopDagPatternMatcher.dummy)
}

class HopDagPatternMatcher(val isLeaf:Boolean = false) {
  val _predicates = new ArrayList[HopPredicate]();
  val _children = new ArrayList[HopDagPatternMatcher]();
  var matchedHops:HashMap[String, Hop] = null
  var variableName:String = null
  
  	
	// Operators that are not yet supported: unary not
	// Operators that are not required: assignment
  def unary_- = new HopDagPatternMatcher().addPredicate("unaryMinus", h => HopRewriteUtils.isBinary(h, OpOp2.MINUS)
				&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0))
				.addChildMatcher(HopDagPatternMatcher.dummy, this)
	def _binary(child2:HopDagPatternMatcher, name:String, op:OpOp2):HopDagPatternMatcher = new HopDagPatternMatcher()
    .addPredicate(name, h => HopRewriteUtils.isBinary(h, op))
		.addChildMatcher(this, child2)
	def _binary(child2:Double, name:String, op:OpOp2):HopDagPatternMatcher = new HopDagPatternMatcher()
    .addPredicate(name, h => HopRewriteUtils.isBinary(h, op) && HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
		.addChildMatcher(this, HopDagPatternMatcher.dummy)
	
	// Matrix-Matrix, Matrix-Vector, Vector-Matrix, ...
  def ^(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "pow", OpOp2.POW)
  def +(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "plus", OpOp2.PLUS)
  def -(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "minus", OpOp2.MINUS)
  def %*%(child2:HopDagPatternMatcher):HopDagPatternMatcher = new HopDagPatternMatcher()
    .addPredicate("matmult", h => HopRewriteUtils.isMatrixMultiply(h))
		.addChildMatcher(this, child2)
	def %/%(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "intdiv", OpOp2.INTDIV)
	def %%(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "modulus", OpOp2.MODULUS)
	def *(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "mult", OpOp2.MULT)
	def /(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "div", OpOp2.DIV)
	def <(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "lt", OpOp2.LESS)
	def >(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "gt", OpOp2.GREATER)
	def <=(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "leq", OpOp2.LESSEQUAL)
	def >=(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "geq", OpOp2.GREATEREQUAL)
	def ==(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "eq", OpOp2.EQUAL)
	def !=(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "neq", OpOp2.NOTEQUAL)
	def &(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "and", OpOp2.AND)
	def |(child2:HopDagPatternMatcher):HopDagPatternMatcher = _binary(child2, "or", OpOp2.OR)
	
	// Matrix-Scalar and Vector-Scalar operators
	def ^(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_pow", OpOp2.POW)
  def +(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_plus", OpOp2.PLUS)
  def -(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_minus", OpOp2.MINUS)
	def %/%(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_intdiv", OpOp2.INTDIV)
	def %%(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_modulus", OpOp2.MODULUS)
	def *(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_mult", OpOp2.MULT)
	def /(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_div", OpOp2.DIV)
	def <(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_lt", OpOp2.LESS)
	def >(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_gt", OpOp2.GREATER)
	def <=(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_leq", OpOp2.LESSEQUAL)
	def >=(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_geq", OpOp2.GREATEREQUAL)
	def ==(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_eq", OpOp2.EQUAL)
	def !=(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_neq", OpOp2.NOTEQUAL)
	def &(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_and", OpOp2.AND)
	def |(child2:Double):HopDagPatternMatcher = _binary(child2, "scalar_or", OpOp2.OR)
	
	// Scalar-Matrix, Scalar-Vector operators are supported via fromDouble
		
  /**
	 * Adds a predicate to the pattern matcher
	 * 
	 * @param name name of the pattern for debugging
	 * @param pred higher order function that takes as an input a hop and returns true if the pattern matches else false
	 * @return this
	 */
  def addPredicate(name:String, pred:Hop => Boolean):HopDagPatternMatcher = {
		_predicates.add(new HopPredicate(name, pred));
		return this;
	}
  
  /**
	 * Add child pattern matcher
	 * @param children list of childer
	 * @return this
	 */
	def addChildMatcher(children: HopDagPatternMatcher*):HopDagPatternMatcher = {
	  children.foreach( c => _children.add(c))
		return this;
	}
	
	/**
	 * Get the matched HOP DAGs
	 * @param varName variable names
	 * @return matched HOP
	 */
	def getMatchedHop(varName:String):Hop = {
		if(matchedHops == null || !matchedHops.containsKey(varName)) {
			throw new RuntimeException("Incorrect usage: the variable " + varName + " is not registered as input.");
		}
		matchedHops.get(varName);
	}
	
	/**
	 * Return the value 
	 * 
	 * @param varName variable name
	 * @return the value of the LiteralOp 
	 */
	def getLiteralValue(varName: String):Double = OptimizerUtils.rEvalSimpleDoubleExpression(getMatchedHop(varName), new HashMap[java.lang.Long, java.lang.Double]());
	
	override def toString() = if(_predicates.size() >= 1) _predicates.get(0).toString() else ""
	
	/**
	 * Match the given HOP DAG
	 * 
	 * @param h root node of the HOP DAG 
	 * @return true if HOP DAG matches
	 */
	 def matches(h:Hop): Boolean = matchHelper(this, h)
	 
	 def matchHelper(root: HopDagPatternMatcher, h:Hop): Boolean = {
		if(h == null || (_children.size() > 0 && h.getInput().size() < _children.size())) {
			if(HopDagPatternMatcher.DEBUG_REWRITES) {
				LOG.info("The expected number of children (" + _children.size() + ") didnot match the number of inputs (" + h.getInput().size() + ") " + this);
			}
			return false;
		}
		for(i  <- 0 until _predicates.size) {
		  val p = _predicates.get(i)
			if(!p._pred(h)) {
				if(HopDagPatternMatcher.DEBUG_REWRITES) {
					LOG.info("The predicate " + p.toString() + " failed for " + Explain.explain(h));
				}
				return false;
			}
		}
		for(i  <- 0 until _children.size) {
		  val child = _children.get(i)
			if(!child.matchHelper(root, h.getInput().get(i))) {
				return false;
			}
		}
		if(isLeaf) {
			if(root.matchedHops == null) {
				root.matchedHops = new HashMap();
			}
			root.matchedHops.put(variableName, h);
		}
		return true;
	}
	
	// Simple helper utilities for adding predicates
	def isScalar():HopDagPatternMatcher = addPredicate("isScalar", h => h.getDataType() == DataType.SCALAR)
	def isMatrix():HopDagPatternMatcher = addPredicate("isMatrix", h => h.getDataType() == DataType.MATRIX)
	def fitsOnGPU(constant:Double) = addPredicate("fitsOnGPU", h => _fitsOnGPU(h, constant))
	def _fitsOnGPU(h:Hop, multiplier:Double):Boolean = {
		val memEst = multiplier*h.getMemEstimate();
		DMLScript.USE_ACCELERATOR && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
}
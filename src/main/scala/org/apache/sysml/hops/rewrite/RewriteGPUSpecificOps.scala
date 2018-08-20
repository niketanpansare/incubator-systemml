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

import java.util.ArrayList
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.hops.rewrite.HopDagPatternMatcher._
import org.apache.sysml.parser.Expression.DataType._

/*
 * -------------------------------------------------------------------------
 * Design documentation for hop rewrite rules that use HopDagPatternMatcher:
 * -------------------------------------------------------------------------
 * 
 * Typical (but not all) hop rewrite rules have following structure:
 * 1. Rules are grouped together in different Java/Scala classes and added in org.apache.sysml.hops.rewrite.ProgramRewriter.
 * 
 * 2. Each rule class inherits from HopRewriteRule and implements rewriteHopDAG method. Other class of rewrite rules are StatementBlockRewriteRule and are not covered by this approach.
 * 
 * 3. The structure of rewriteHopDAG is common across HopRewriteRule subclasses and usually have following pattern:
 *  if(root of the given HOP DAG matches certain pattern) {
 *    HopRewriteUtils.rewireAllParentChildReferences(root, newRoot)
 *  }
 *  else root
 * 
 * 4. To avoid redundancy, the above logic is implemented in the abstract class HopRewriteRuleWithPatternMatcher:
 *  val patternMatchers = getPatternMatchers()
 *  for(i <- 0 until patternMatchers.size) {
 *    val matcher = patternMatchers.get(i)
 *    root = if(matcher.getMatcher.matches(root)) matcher.getReplacer()(root) else root
 *  }
 * 
 * 5. The developer has to inherit from HopRewriteRuleWithPatternMatcher that implements the above logic
 * and write code for getPatternMatchers() that returns ArrayList[HopDagPatternReplacementPair]  
 * 
 * 6. Since the HOP pattern donot change during execution, it is convenient to implement them into a static variable: 
 * val patternMatcher = new ArrayList[HopDagPatternReplacementPair]
 * 
 * 7. The replacement part in each entry of patternMatcher invokes the helper methods in HopRewriteUtils to create a newRoot. For example: HopRewriteUtils.createDnnOp
 * 
 * 8. The matcher part in each entry of patternMatcher uses the DSL implemented in HopDagPatternMatcher to improve readability.
 * - The DSL mentioned above follows DML syntax that makes it convenient for an external contributer to understand and modify the HOP rewrites: 
 * bias_multiply(bias_add(X, -mean), one / sqrt(variance + eps)) 
 * - It is important to note that the developer has to add the same scoping rules as SystemML and use brackets to avoid ambiguity.
 * (X - (mu*v_prev)) + (onePlusMu*v) rather than (X - mu*v_prev + onePlusMu*v)
 * - To create a newRoot HOP, it is important to have a mechanism to extract leaves of the matched pattern. This is implemented
 * by using leaf() method.
 * (X - (mu*leaf("v_prev", MATRIX))) + (onePlusMu*leaf("v", MATRIX))
 * - Often, it is important to create a new HOP only if it it can fit into memory. For GPU, one can use the fitsOnGPU(multiplier) helper method:
 * (X.fitsOnGPU(4) - (mu*leaf("v_prev", MATRIX))) + (onePlusMu*leaf("v", MATRIX))
 * 
 */
object RewriteGPUSpecificOps {
  // Helper methods for commonly used matrix/scalar variables and constants:
  private def X = leaf("X", MATRIX)
  private def mean = leaf("mean", MATRIX)
  private def variance = leaf("var", MATRIX) // var is reserved keyword in Scala
  private def eps = leaf("eps", SCALAR)
  private def C = leaf("C", SCALAR)
  private def HW = leaf("HW", SCALAR)
  private def mu = leaf("mu", SCALAR)
  private def onePlusMu = leaf("onePlusMu", SCALAR)
  private def oneMinusMu = leaf("oneMinusMu", SCALAR)
  private def one = fromDouble(1.0)
  
  private val patternMatcher = new ArrayList[HopDagPatternReplacementPair]
  // -------------------------------------------------------------------------------------------
	// Pattern 1:
	// subgrp_vars = matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
	// batchNormUpdatedVar = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
  private def _subgrp_vars = matrix(colVars(X.fitsOnGPU(2) * leaf("varConst1", SCALAR)), rows=C, cols=HW)
  val batchNormUpdatedVar = rowMeans(_subgrp_vars) + (rowVars(leaf("subgrp_means", MATRIX))*leaf("varConst2", SCALAR))
  patternMatcher.add(new HopDagPatternReplacementPair(batchNormUpdatedVar, root => {
    val HW = batchNormUpdatedVar.getLiteralValue("HW");
		if(batchNormUpdatedVar.getLiteralValue("varConst2") == ((HW-1)/HW)) {
      LOG.debug("Applied batchNormUpdatedVar rewrite.");
      HopRewriteUtils.rewireAllParentChildReferences(root,
        HopRewriteUtils.createDnnOp(batchNormUpdatedVar, OpOpDnn.UPDATE_EMA_VAR, "subgrp_means", "X", "C", "HW", "varConst1"))
		}
		else root
  }))
  
  // Pattern 2:
  // norm = bias_multiply(bias_add(X, -mean), 1/sqrt(variance+eps))
  // batchNormTest = bias_add(bias_multiply(norm, gamma), beta)
  private def _norm = bias_multiply(bias_add(X.fitsOnGPU(3), -mean), one / sqrt(variance + eps))
  val batchNormTest = bias_add(bias_multiply(_norm, leaf("gamma", MATRIX)), leaf("beta", MATRIX))
  patternMatcher.add(new HopDagPatternReplacementPair(batchNormTest, root => {
    LOG.debug("Applied batchNormTest rewrite.")
    HopRewriteUtils.rewireAllParentChildReferences(root,
        HopRewriteUtils.createDnnOp(batchNormTest, OpOpDnn.BATCH_NORM2D_TEST, "X", "gamma", "beta", "mean", "var", "eps"))
  }))
  
  // Pattern 3:
	// rowSums(matrix(colSums(X), rows=C, cols=HW))
  val channelSums = rowSums(matrix(colSums(X.fitsOnGPU(2)), rows=C, cols=HW))
  patternMatcher.add(new HopDagPatternReplacementPair(channelSums, root => {
    LOG.debug("Applied channelSums rewrite.")
    HopRewriteUtils.rewireAllParentChildReferences(root,
        HopRewriteUtils.createDnnOp(channelSums, OpOpDnn.CHANNEL_SUMS, "X", "C", "HW"))
  }))
  
  // Pattern 4:
	// updateNesterovX = (X - (mu*v_prev)) + (1+mu)*v
  val updateNesterovX = (X.fitsOnGPU(4) - (mu*leaf("v_prev", MATRIX))) + (onePlusMu*leaf("v", MATRIX))
  patternMatcher.add(new HopDagPatternReplacementPair(updateNesterovX, root => {
    var ret = root
    if((1+updateNesterovX.getLiteralValue("mu")) == updateNesterovX.getLiteralValue("onePlusMu")) {
      val X = updateNesterovX.getMatchedHop("X");
			val v = updateNesterovX.getMatchedHop("v");
			val v_prev = updateNesterovX.getMatchedHop("v_prev");
			if(hasSameDimensions(X, v) && hasSameDimensions(X, v_prev)) {
				LOG.debug("Applied updateNesterovX rewrite.");
				ret = HopRewriteUtils.rewireAllParentChildReferences(root, 
				    HopRewriteUtils.createDnnOp(updateNesterovX, OpOpDnn.UPDATE_NESTEROV_X, "X", "v", "v_prev", "mu"));
			}
    }
    ret
  }))
  
  // Pattern 5:
	// reshapeColMeans = matrix(colMeans(X), rows=C, cols=Hin*Win)
	// This avoids unnecessary copy by the reshape operator
  val reshapeColMeans = matrix(colMeans(X.fitsOnGPU(2)), rows=C, cols=HW)
  patternMatcher.add(new HopDagPatternReplacementPair(reshapeColMeans, root => {
    LOG.debug("Applied reshapeColMeans rewrite.")
    HopRewriteUtils.rewireAllParentChildReferences(root,
        HopRewriteUtils.createDnnOp(reshapeColMeans, OpOpDnn.RESHAPE_COLMEANS, "X", "C", "HW"))
  }))
  
  // Pattern 6:
	// updateEMA = (mu*ema_mean) + ((1-mu)*mean)
  val updateEMA = (mu*leaf("ema_mean", MATRIX)) + (oneMinusMu*mean) 
  patternMatcher.add(new HopDagPatternReplacementPair(updateEMA,  root => {
    if((1-updateEMA.getLiteralValue("mu")) == updateEMA.getLiteralValue("oneMinusMu")) {
      LOG.debug("Applied updateEMA rewrite.")
      HopRewriteUtils.rewireAllParentChildReferences(root,
        HopRewriteUtils.createDnnOp(updateEMA, OpOpDnn.UPDATE_EMA, "ema_mean", "mean", "mu"))
    }
    else root
  }))
  
  
  // Pattern 7:
	// invVar = 1/sqrt(var+epsilon)
  val invVar = one / sqrt(variance + eps)
  patternMatcher.add(new HopDagPatternReplacementPair(updateEMA,  root => {
    LOG.debug("Applied invVar rewrite.")
    HopRewriteUtils.rewireAllParentChildReferences(root,
      HopRewriteUtils.createDnnOp(invVar, OpOpDnn.INV_VAR, "var", "eps"))
  }))
  // -------------------------------------------------------------------------------------------
  
  def hasSameDimensions(x:Hop, y:Hop):Boolean = {
		return x.dimsKnown() && y.dimsKnown() && (x.getDim1() == y.getDim1()) && (x.getDim2() == y.getDim2());
	}
}


class RewriteGPUSpecificOps extends HopRewriteRuleWithPatternMatcher {
  override def getPatternMatchers():ArrayList[HopDagPatternReplacementPair] = RewriteGPUSpecificOps.patternMatcher
}
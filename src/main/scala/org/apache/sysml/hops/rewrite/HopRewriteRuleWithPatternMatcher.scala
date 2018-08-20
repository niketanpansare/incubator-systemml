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
import org.apache.sysml.hops.Hop

class HopDagPatternReplacementPair(pattern:HopDagPatternMatcher, replacement:Hop => Hop) {
  def getMatcher():HopDagPatternMatcher = pattern
  def getReplacer():Hop => Hop = replacement
}

abstract class HopRewriteRuleWithPatternMatcher extends HopRewriteRule {
  def getPatternMatchers():ArrayList[HopDagPatternReplacementPair]
  override def rewriteHopDAGs(roots:ArrayList[Hop], state:ProgramRewriteStatus): ArrayList[Hop] = {
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for(i <- 0 until roots.size)
			rule_helper(roots, roots.get(i), false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for(i <- 0 until roots.size)
			rule_helper(roots, roots.get(i), true );
		Hop.resetVisitStatus(roots, true);
		
		return roots;
	}
  override def rewriteHopDAG(root:Hop, state:ProgramRewriteStatus): Hop = {
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		rule_helper(null, root, false );
		
		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		rule_helper(null, root, true );
		
		return root;
	}
  def rule_helper(roots:ArrayList[Hop], hop:Hop, descendFirst:Boolean):Unit = {
		if(hop.isVisited())
			return;
		
		//recursively process children
		for(i <- 0 until hop.getInput().size) {
			var root = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_helper(roots, root, descendFirst); //see below
			
			val patternMatchers = getPatternMatchers()
			for(i <- 0 until patternMatchers.size) {
			  val matcher = patternMatchers.get(i)
			  root = if(matcher.getMatcher.matches(root)) matcher.getReplacer()(root) else root
			}
	
			if( !descendFirst )
				rule_helper(roots, root, descendFirst);
		}
		hop.setVisited();
	}
}
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
package org.apache.sysml.parser.dml;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;

public class InlineableMethods {
	ArrayList<String> _variables;
	final String _body;
	final String _fnName;
	
	public InlineableMethods(String fnName, String body, ArrayList<String> variables) {
		_fnName = fnName;
		_body = body;
		_variables = variables;
		_variables.sort(Comparator.comparing(String::length).reversed());
	}
	
	public ArrayList<String> getLocalVariables() {
		return _variables;
	}
	
	public String getInlinedDML(int callerID, HashMap<String, String> actualArguments) {
		String ret = _body;
		for(String var : _variables) {
			String originalVarName = var.substring(InlineHelper.ARG_PREFIX.length());
			if(actualArguments.containsKey(originalVarName)) {
				ret = ret.replaceAll(var, actualArguments.get(originalVarName));
			}
			else {
				// internal argument
				ret = ret.replaceAll(var, LOCAL_ARG_PREFIX + _fnName + "_" + callerID + "_" + originalVarName);
			}
		}
		return ret;
	}
	
	static final String LOCAL_ARG_PREFIX;
	static {
//		Random rand = new Random();
//		LOCAL_ARG_PREFIX = "LOCAL_" + Math.abs(rand.nextLong()) + "_";
		LOCAL_ARG_PREFIX = "LOCAL_";
	}
}

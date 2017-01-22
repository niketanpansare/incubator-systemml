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
package org.apache.sysml.udf.lib;

import java.util.ArrayList;

public interface Py4JCallbackInterface {
	public ArrayList<Object> performOperation(Object x1);
	public ArrayList<Object> performOperation(Object x1, Object x2);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5, Object x6);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5, Object x6, Object x7);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5, Object x6, Object x7, Object x8);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5, Object x6, Object x7, Object x8, Object x9);
	public ArrayList<Object> performOperation(Object x1, Object x2, Object x3, Object x4, Object x5, Object x6, Object x7, Object x8, Object x9, Object x10);
}

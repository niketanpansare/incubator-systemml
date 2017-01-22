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
package org.apache.sysml.api

import org.apache.sysml.udf.FunctionParameter
import org.apache.sysml.udf.lib.Py4JCallbackInterface
import org.apache.sysml.udf.lib.GenericFunction
import java.util.ArrayList

class Py4JCallbackImpl(name: String, callbackObj:Py4JCallbackInterface, funcSign:ArrayList[String], ml:mlcontext.MLContext) extends Function0[Array[FunctionParameter]] {
  
  var udf:GenericFunction = null
  
  // Add signature to global function
  val funcSignature:Array[String] = new Array[String](funcSign.size)
  for(i <- 0 until funcSign.size) {
    funcSignature(i) = funcSign.get(i)
  }
  org.apache.sysml.api.ExternalUDFRegistration.fnSignatureMapping.put(name, funcSignature)
  org.apache.sysml.api.ExternalUDFRegistration.fnMapping.put(name, this)
  ml.udf.scriptHeader.append(name + " = externalFunction(")
  for(i <- 0 until funcSignature.size-1) {
      if(i != 0)
          ml.udf.scriptHeader.append(", ")
      ml.udf.scriptHeader.append(funcSignature(i) + " X" + i)
  }
  ml.udf.scriptHeader.append(") return (")
  // TODO: Multi-return argument
  ml.udf.scriptHeader.append(funcSignature(funcSignature.size-1) + " Y")
  ml.udf.scriptHeader.append(") implemented in (classname=\"org.apache.sysml.udf.lib.GenericFunction\", exectype=\"mem\");\n")
  
  
  def apply(): Array[FunctionParameter] = {
    udf = ExternalUDFRegistration.udfMapping.get(name)
    funcSignature.length match {
      case 2 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0)))
      case 3 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1)))
      case 4 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2)))
      case 5 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3)))
      case 6 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3), get(4)))
      case 7 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3), get(4), 
          get(5)))
      case 8 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3), get(4), 
          get(5), get(6)))
      case 9 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3), get(4), 
          get(5), get(6), get(7)))
      case 10 => return ExternalUDFRegistration.convertReturnToOutput(callbackObj.performOperation(get(0), get(1), get(2), get(3), get(4), 
          get(5), get(6), get(7), get(8)))
    }
    
    throw new RuntimeException("Unsupported function signature")
  }
  
  def get(pos:Int) = udf.getPythonInput(funcSignature(pos), pos)
  
}
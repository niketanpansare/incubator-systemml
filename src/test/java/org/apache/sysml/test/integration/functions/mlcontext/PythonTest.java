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

package org.apache.sysml.test.integration.functions.mlcontext;

import java.io.File;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.junit.Test;
import org.apache.spark.deploy.PythonRunner;

public class PythonTest extends AutomatedTestBase {

	@Override
	public void setUp() { }

	@Test
	public void runTestPy() throws DMLRuntimeException {
		// Remember to set SPARK_HOME and also 
		// export PYTHONPATH=$SPARK_HOME/python/lib/py4j-???-src.zip:$PYTHONPATH
		
		String sparkHome = System.getenv("SPARK_HOME");
		if(sparkHome == null || !(new File(sparkHome)).exists()) {
			throw new DMLRuntimeException("SPARK_HOME environment variable needs to be set");
		}
		
		String pythonPath = System.getenv("PYTHONPATH");
		
		String [] inputPythonFiles = new String[2];
		String libDir = sparkHome + File.separator + "python" + File.separator + "lib";
		File[] files = new File(libDir).listFiles();
		String py4jFile = null;
		String pyspark = null;
		for (File file : files) {
			if (file.isFile() && file.getName().startsWith("py4j")) {
				py4jFile = file.getAbsolutePath();
			}
			else if (file.isFile() && file.getName().startsWith("pyspark")) {
				pyspark = file.getAbsolutePath();
			}
		}
		
		if(py4jFile == null || pyspark == null) {
			throw new DMLRuntimeException("Expected py4j-*.zip and pyspark.zip in directory:" + libDir);
		}
		
		if(pythonPath == null || !pythonPath.contains("py4j")) {
			throw new DMLRuntimeException("Include \""+ py4jFile + "\" in PYTHONPATH environment variable");
		}
		System.out.println("Using py4j:" + py4jFile + " and pyspark:" + pyspark);
		inputPythonFiles[0] = "src/main/java/org/apache/sysml/api/python/test.py";
		inputPythonFiles[1] = "src/main/java/org/apache/sysml/api/python/SystemML.py," + py4jFile + "," + pyspark;
		PythonRunner.main(inputPythonFiles);
	}
}

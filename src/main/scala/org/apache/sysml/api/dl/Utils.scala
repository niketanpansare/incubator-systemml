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
package org.apache.sysml.api.dl
import scala.collection.JavaConversions._
import java.util.ArrayList
import java.util.HashSet
import scala.collection.mutable.Stack
import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import org.apache.sysml.parser.LanguageException;
import com.google.protobuf.TextFormat;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import caffe.Caffe.SolverParameter;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

object Utils {
  def shouldVisit(layer:LayerParameter, visited:HashSet[String]):Boolean = {
	  val bottom = layer.getBottomList
	  if(layer.getBottomCount == 0 ||
	      (layer.getBottomCount == 1 && bottom(0).compareTo(layer.getName) == 0)) {
	    return true
	  }
	  else {
	    for(b <- bottom) {
	      if(!visited.contains(b.compareTo(layer.getName))) {
	        return false
	      }
	    }
	    return true
	  }
	}
	def getTopologicalSortedLayers(net:NetParameter, layers:Map[String, (LayerParameter, Int)]): ArrayList[String] = {
	  val visited:HashSet[String] = new HashSet[String]()
	  val ret:ArrayList[String] = new ArrayList[String]()
	  while(visited.size != layers.size) {
	    var atleastOneVisited = false
	    for(l <- net.getLayerList) {
	      if(shouldVisit(l, visited)) {
	        visited.add(l.getName)
	        ret.add(l.getName)
	        atleastOneVisited = true
	      }
	    }
	    if(!atleastOneVisited && visited.size < layers.size) {
	      throw new LanguageException("Possible cycle")
	    }
	  }
	  ret
	}
	
	// --------------------------------------------------------------
	// Caffe utility functions
	def readCaffeNet(solver:SolverParameter):NetParameter = {
		val reader:InputStreamReader = getInputStreamReader(solver.getNet()); // TODO:
  	val builder:NetParameter.Builder =  NetParameter.newBuilder();
  	TextFormat.merge(reader, builder);
  	return builder.build();
	}
	
	def readCaffeSolver(solverFilePath:String):SolverParameter = {
		val reader = getInputStreamReader(solverFilePath);
		val builder =  SolverParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}
	
	// --------------------------------------------------------------
	// File IO utility functions
	def getInputStreamReader(filePath:String ):InputStreamReader = {
		//read solver script from file
		if(filePath == null)
			throw new LanguageException("file path was not specified!");
		if(filePath.startsWith("hdfs:")  || filePath.startsWith("gpfs:")) { 
			if( !LocalFileUtils.validateExternalFilename(filePath, true) )
				throw new LanguageException("Invalid (non-trustworthy) hdfs filename.");
			val fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
			return new InputStreamReader(fs.open(new Path(filePath)));
		}
		else { 
			if( !LocalFileUtils.validateExternalFilename(filePath, false) )
				throw new LanguageException("Invalid (non-trustworthy) local filename.");
			return new InputStreamReader(new FileInputStream(new File(filePath)), "ASCII");
		}
	}
	// --------------------------------------------------------------
}
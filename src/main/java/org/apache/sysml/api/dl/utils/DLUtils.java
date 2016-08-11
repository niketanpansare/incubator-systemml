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
package org.apache.sysml.api.dl.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.layer.AccuracyLayer;
import org.apache.sysml.api.dl.layer.ConvolutionLayer;
import org.apache.sysml.api.dl.layer.DataLayer;
import org.apache.sysml.api.dl.layer.DropoutLayer;
import org.apache.sysml.api.dl.layer.InnerProductLayer;
import org.apache.sysml.api.dl.layer.Layer;
import org.apache.sysml.api.dl.layer.PoolingLayer;
import org.apache.sysml.api.dl.layer.ReLULayer;
import org.apache.sysml.api.dl.layer.SoftmaxWithLossLayer;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.LocalFileUtils;

import caffe.Caffe.LayerParameter;
import caffe.Caffe.NetParameter;
import caffe.Caffe.SolverParameter;

import com.google.protobuf.TextFormat;

public class DLUtils {
	
	public static ArrayList<Layer> findLayerWithTop(String topName) {
		ArrayList<Layer> ret = new ArrayList<Layer>();
		for(Layer l : Barista.dmlNet) {
			if(l.param != null && l.param.getTopCount() > 0) {
				for(String top : l.param.getTopList()) {
					if(top.equalsIgnoreCase(topName)) {
						ret.add(l);
						break;
					}
				}
			}
		}
		return ret;
	}

	public static NetParameter readCaffeNet(SolverParameter solver) throws IOException, LanguageException {
		InputStreamReader reader = getInputStreamReader(solver.getNet());
		NetParameter.Builder builder =  NetParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}

	public static SolverParameter readCaffeSolver(String solverFilePath) throws IOException, LanguageException {
		InputStreamReader reader = getInputStreamReader(solverFilePath);
		SolverParameter.Builder builder =  SolverParameter.newBuilder();
		TextFormat.merge(reader, builder);
		return builder.build();
	}

	public static Layer readLayer(LayerParameter param) throws DMLRuntimeException {
		if(param.getType().equals("Data")) {
			return new DataLayer(param);
		}
		else if(param.getType().equals("Convolution")) {
			return new ConvolutionLayer(param);
		}
		else if(param.getType().equals("Pooling")) {
			return new PoolingLayer(param);
		}
		else if(param.getType().equals("InnerProduct")) {
			return new InnerProductLayer(param);
		}
		else if(param.getType().equals("ReLU")) {
			return new ReLULayer(param);
		}
		else if(param.getType().equals("Accuracy")) {
			return new AccuracyLayer(param);
		}
		else if(param.getType().equals("SoftmaxWithLoss")) {
			return new SoftmaxWithLossLayer(param);
		}
		else if(param.getType().equals("Dropout")) {
			return new DropoutLayer(param);
		}
		throw new DMLRuntimeException("A layer of type " + param.getType() + " is not implemented.");
	}
	
	public static String getP_DML(String H, int pad_h, int R, int stride_h) {
		return MathUtils.toInt(MathUtils.scalarAddition(
				MathUtils.scalarDivision(
				MathUtils.scalarSubtraction(
						MathUtils.scalarAddition(H, MathUtils.scalarMultiply("2", "" + pad_h)),
						"" + R),
				"" + stride_h), "1"));
	}
	
	public static String getQ_DML(String W, int pad_w, int S, int stride_w) {
		return MathUtils.toInt(MathUtils.scalarAddition(
				MathUtils.scalarDivision(
				MathUtils.scalarSubtraction(
						MathUtils.scalarAddition(W, MathUtils.scalarMultiply("2", "" + pad_w)),
						"" + S),
				"" + stride_w), "1"));
	}
	
	private static InputStreamReader getInputStreamReader( String filePath ) 
			throws IOException, LanguageException
		{
			//read solver script from file
			if(filePath == null)
				throw new LanguageException("file path was not specified!");
			
			try 
			{
				//read from hdfs or gpfs file system
				if(    filePath.startsWith("hdfs:") 
					|| filePath.startsWith("gpfs:") ) 
				{ 
					if( !LocalFileUtils.validateExternalFilename(filePath, true) )
						throw new LanguageException("Invalid (non-trustworthy) hdfs filename.");
					FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
					Path scriptPath = new Path(filePath);
					return new InputStreamReader(fs.open(scriptPath));
				}
				// from local file system
				else 
				{ 
					if( !LocalFileUtils.validateExternalFilename(filePath, false) )
						throw new LanguageException("Invalid (non-trustworthy) local filename.");
					// in = new BufferedReader(new FileReader(script));
					return new InputStreamReader(new FileInputStream(new File(filePath)), "ASCII");
				}
				
			}
			catch (IOException ex)
			{
				throw ex;
			}
					
		}
	
}
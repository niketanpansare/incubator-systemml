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
package org.apache.sysml.api.dl.layer;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class DataLayer extends Layer {
	int currentIndex = 1;
	public String labelVar;
	
	public DataLayer(LayerParameter param) {
		super(param, "Xb");
	}

	@Override
	public String getSetupDML() {
		Barista.batchSize = param.getDataParam().getBatchSize();
		
		String zScoring = "# z-scoring features\n" +
				"applyZScore = ifdef($applyZScore, true);\n" +
				"if(applyZScore) {" +
				"\t#means = colSums(X)/num_images\n" +
				"\t#stds = sqrt((colSums(X^2)/num_images - means^2)*num_images/(num_images-1)) + 1e-17\n" +
				"\tmeans = 255/2;\n" +
				"\tstds = 255;\n" +
				"\tX = (X - means)/stds;\n" +
				"}\n";
		
		
		String setup = "# Assumption: (X, y) split into two files\n" +
				"X = read(\"" + param.getDataParam().getSource() + ".X\");\n" +
				"y = read(\"" + param.getDataParam().getSource() + ".y\");\n" +
				"classes = " + Barista.numClasses + ";\n" +
				"BATCH_SIZE = " + Barista.batchSize + ";\n" +
				"num_images = nrow(y);\n" +
				zScoring +
				"Xv = X[1:" + Barista.numValidation + ",];\n" +
				"yv = y[1:" + Barista.numValidation + ",];\n" +
				"num_images_validation = 5000;\n" +
				"X = X[" + (Barista.numValidation+1) + ":num_images];\n" +
				"y = y[" + (Barista.numValidation+1) + ":num_images];\n" +
				"Y = table(seq(1,num_images,1), y+1, num_images, classes);\n" +
				"beg = 1;\n" +
				"num_iters_per_epoch = ceil(num_images/BATCH_SIZE); \n";
		return setup;
	}

	@Override
	public String getForwardDML() {
		labelVar = "Yb";
		String batch = "end = beg + BATCH_SIZE - 1;\n"
				+ "\t" + "if(end > n) end = n;\n\n"
				+ "\t" + "# Pulling out the batch\n"
				+ "\t" + "Xb = X[beg:end,]\n"
				+ "\t" + "n = nrow(Xb);\n"
				+ "\t" +  labelVar + " = Y[beg:end,];\n\n"
				+ "\t" + "beg = beg + BATCH_SIZE;\n" 
				+ "\t" + "if(beg > num_images) beg = 1;\n"
				+ "\t" + "if(iter %% num_iters_per_epoch == 0) step_size = step_size * decay";
		return batch;
	}

	@Override
	public String getBackwardDML() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getFinalizeDML() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		output_shape.clear();
		output_shape.add("n"); // TODO:
		// output_shape.add("" + Barista.batchSize);
		output_shape.add("" + Barista.numChannelsOfInputData);
		output_shape.add("" + Barista.inputHeight);
		output_shape.add("" + Barista.inputWidth);
	}

}
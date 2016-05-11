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
import org.apache.sysml.api.dl.utils.MathUtils;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class DataLayer extends Layer {
	int currentIndex = 1;
	public String labelVar;
	
	public DataLayer(LayerParameter param) {
		super(param, "X" + Barista.trainingVarsuffix);
		labelVar = "y";
	}

	@Override
	public String getSetupDML() {
		Barista.batchSize = param.getDataParam().getBatchSize();
		
		String zScoring = "# z-scoring features\n" +
				"applyZScore = ifdef($applyZScore, TRUE);\n" +
				"if(applyZScore) {\n" +
				"\t#means = colSums(X)/num_images\n" +
				"\t#stds = sqrt((colSums(X^2)/num_images - means^2)*num_images/(num_images-1)) + 1e-17\n" +
				"\tmeans = 255/2;\n" +
				"\tstds = 255;\n" +
				"\tX = (X - means)/stds;\n" +
				"}\n";
		
		
		String setup = "# Assumption: (X, " + labelVar + ") split into two files\n" +
				"X = read(\"" + param.getDataParam().getSource() + ".X\");\n" +
				labelVar + " = read(\"" + param.getDataParam().getSource() + "." + labelVar + "\");\n" +
				"classes = " + Barista.numClasses + ";\n" +
				"BATCH_SIZE = " + Barista.batchSize + ";\n" +
				"num_images = nrow(y);\n" +
				zScoring +
				"# Split training data into training and validation set\n" +
				"X" + Barista.validationVarsuffix + " = X[1:" + Barista.numValidation + ",];\n" +
				labelVar + Barista.validationVarsuffix + " = " + labelVar + "[1:" + Barista.numValidation + ",];\n" +
				"X = X[" + MathUtils.scalarAddition(Barista.numValidation, "1") + ":num_images,];\n" +
				 labelVar + " = "+ labelVar +"[" + MathUtils.scalarAddition(Barista.numValidation, "1") + ":num_images,];\n"+ 
				 "oneHotEncoded_" + labelVar + " = table(seq(1,num_images,1), " + labelVar + "+1, num_images, classes);\n";
		return setup;
	}

	@Override
	public String getForwardDML() {
		
		String batch = "end = beg + BATCH_SIZE - 1;\n"
				+ "\t" + "if(end > num_images) end = num_images;\n\n"
				+ "\t" + "# Pulling out the batch\n"
				+ "\t" + outputVar + " = X[beg:end,];\n"
				+ "\t" + "n = nrow(" + outputVar + ");\n"
				+ "\t" + "oneHotEncoded_" + labelVar + Barista.trainingVarsuffix  + " = " + "oneHotEncoded_" + labelVar + "[beg:end,];\n\n"
				+ "\t" + "beg = beg + BATCH_SIZE;\n" 
				+ "\t" + "if(beg > num_images) beg = 1;\n";
				// + "\t" + "#if(iter %% num_iters_per_epoch == 0) step_size = step_size * decay";
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
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
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
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
	public void generateSetupDML(StringBuilder dmlScript) throws DMLRuntimeException {
		Barista.BATCH_SIZE = param.getDataParam().getBatchSize();
		
		printSetupHeader(dmlScript);
		dmlScript.append("# Assumption: (X, " + labelVar + ") split into two files\n");
		dmlScript.append(assign("X", read(inQuotes(param.getDataParam().getSource() + ".X"))));
		dmlScript.append(assign(labelVar, read(inQuotes(param.getDataParam().getSource() + "." + labelVar))));
		dmlScript.append(assign("classes", Barista.numClasses));
		dmlScript.append(assign("BATCH_SIZE", ""+Barista.BATCH_SIZE));
		dmlScript.append(assign("num_images", nrow(labelVar)));
		
		// Z-Scoring
		dmlScript.append("# z-scoring features\n");
		dmlScript.append(assign("applyZScore", ifdef("applyZScore", "TRUE")));
		dmlScript.append(ifLoop("applyZScore"));
		
		TabbedStringBuilder tDmlScript = new TabbedStringBuilder(dmlScript, 1);
		tDmlScript.append("#means = colSums(X)/num_images\n");
		tDmlScript.append("#stds = sqrt((colSums(X^2)/num_images - means^2)*num_images/(num_images-1)) + 1e-17\n");
		tDmlScript.append(assign("means", "255/2"));
		tDmlScript.append(assign("stds", "255"));
		tDmlScript.append(assign("X", divide(subtract("X", "means", true), "stds") ));
		dmlScript.append("}\n");
		
		dmlScript.append("# Split training data into training and validation set\n");
		dmlScript.append(assign("X"+Barista.validationVarsuffix, "X[1:" + Barista.numValidation + ",]"));		 
		dmlScript.append(assign(labelVar + Barista.validationVarsuffix, labelVar + "[1:" + Barista.numValidation + ",]"));	
		dmlScript.append(assign("X", "X[" + MathUtils.scalarAddition(Barista.numValidation, "1") + ":num_images,]"));
		dmlScript.append(assign(labelVar, labelVar +"[" + MathUtils.scalarAddition(Barista.numValidation, "1") + ":num_images,]"));
		dmlScript.append(assign("num_images", nrow(labelVar)));
		dmlScript.append(assign("oneHotEncoded_" + labelVar, 
				table(seq(1, "num_images", 1), add(labelVar, "1"), "num_images", "classes")));
	}

	@Override
	public void generateForwardDML(TabbedStringBuilder dmlScript) {
		printForwardHeader(dmlScript);
		dmlScript.append(assign("end", subtract(add("beg", "BATCH_SIZE"), "1")));
		dmlScript.append("if(end > num_images) end = num_images;\n\n");
		dmlScript.append("# Pulling out the batch\n");
		dmlScript.append(assign(outputVar, "X[beg:end,]"));
		dmlScript.append(assign("n", nrow(outputVar)));
		dmlScript.append(assign("oneHotEncoded_" + labelVar + Barista.trainingVarsuffix, 
								"oneHotEncoded_" + labelVar + "[beg:end,]"));
		dmlScript.append(assign("beg", add("beg", "BATCH_SIZE")));
		dmlScript.append("if(beg > num_images) beg = 1;\n");
	}

	@Override
	public void generateBackwardDML(TabbedStringBuilder dmlScript) throws DMLRuntimeException {
	}

	@Override
	public String generateFinalizeDML() {
		return null;
	}

	@Override
	public void updateOutputShape() throws DMLRuntimeException {
		output_shape.clear();
		output_shape.add("n"); 
		output_shape.add("" + Barista.numChannelsOfInputData);
		output_shape.add("" + Barista.inputHeight);
		output_shape.add("" + Barista.inputWidth);
	}

}
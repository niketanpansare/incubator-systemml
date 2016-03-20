package org.apache.sysml.api.dl.layers;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public class DataLayer extends Layer {
	int currentIndex = 1;
	
	public DataLayer(LayerParameter param) {
		super(param, "X_batch");
		System.out.println("WARN: Only supporting binary block formats for now with no transformation !!");
	}

	@Override
	public String getSetupDML() {
		Barista.batchSize = param.getDataParam().getBatchSize();
		String setup = "X = read(\"" + param.getDataParam().getSource()  + "\") # WARN: Only supporting binary block formats for now with no transformation" + "\n" +
				"batch_size = " + Barista.batchSize + "\n" +
				"n = nrow(X)\n" +
				"m = ncol(X)\n" +
				"max_batch_iterations = 3\n";
		return setup;
	}

	@Override
	public String getForwardDML() {
		String batch = "beg = ((iter %/% max_batch_iterations) * batch_size) %% n + 1\n" +
						"\t" + "end = beg + batch_size - 1\n" + 
						"\t" + "if(1==1){\n" +
						"\t\t" + "X_batch = X[beg:end,]\n" +
						"\t" + "}\n" +
						"\t" + "print(\"NEW BATCH beg=\" + beg + \" end=\" + end)\n" +
						"\t" + "n_batch = nrow(X_batch)\n";
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
		output_shape.add("n_batch"); // TODO:
		// output_shape.add("" + Barista.batchSize);
		output_shape.add("" + Barista.numChannelsOfInputData);
		output_shape.add("" + Barista.inputHeight);
		output_shape.add("" + Barista.inputWidth);
	}

}

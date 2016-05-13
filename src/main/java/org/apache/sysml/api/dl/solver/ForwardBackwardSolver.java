package org.apache.sysml.api.dl.solver;

import java.util.ArrayList;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.layer.DataLayer;
import org.apache.sysml.api.dl.layer.Layer;
import org.apache.sysml.api.dl.layer.SoftmaxWithLossLayer;
import org.apache.sysml.api.dl.utils.MathUtils;
import org.apache.sysml.api.dl.utils.TabbedStringBuilder;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.SolverParameter;

public class ForwardBackwardSolver {
	
	public String getForwardBackwardDML(ArrayList<Layer> dmlNet, SolverParameter solver) throws DMLRuntimeException {
		StringBuilder dmlScript = new StringBuilder();
		
		if(solver.hasType() && !solver.getType().equalsIgnoreCase("SGD")) {
			throw new DMLRuntimeException("Only SGD solver is implemented in this version.");
		}
		
		// TODO: Adding defaults for MNIST: Remove it later
		dmlScript.append(Barista.numChannelsOfInputData + " = ifdef($" + Barista.numChannelsOfInputData + ", 1);\n");
		dmlScript.append(Barista.inputHeight + " = ifdef($" + Barista.inputHeight + ", 28);\n");
		dmlScript.append(Barista.inputWidth + " = ifdef($" + Barista.inputWidth + ", 28);\n");
		dmlScript.append(Barista.numClasses + " = ifdef($" + Barista.numClasses + ", 10);\n");
		dmlScript.append(Barista.numValidation + " = ifdef($" + Barista.numValidation + ", 5000);\n");
		
		if(solver.hasMomentum() && solver.getMomentum() != 0) {
			Barista.USE_MOMENTUM = true;
		}
		
		if(solver.hasWeightDecay()) {
			dmlScript.append("# Regularization constant\n");
			dmlScript.append("lambda = " +  MathUtils.toDouble(solver.getWeightDecay()) + ";\n");
		}
		if(Barista.USE_MOMENTUM) {
			dmlScript.append("# Momentum\n");
			dmlScript.append("mu = " +  MathUtils.toDouble(solver.getMomentum()) + ";\n");
		}
		if(solver.hasGamma())
			dmlScript.append("gamma = " + MathUtils.toDouble(solver.getGamma()) + ";\n");
		else
			throw new DMLRuntimeException("Expected gamma parameter in the solver");
		
		for(Layer layer : dmlNet) {
			layer.generateSetupDML(dmlScript); 
			layer.updateOutputShape();
		}
		dmlScript.append("\n");
		dmlScript.append(Barista.step_size + " = " + solver.getBaseLr() + "; # from solver's base_lr parameter\n");
		dmlScript.append("iter = 0;\n");
		dmlScript.append("beg = 1;\n");
		dmlScript.append("maxiter = " + solver.getMaxIter() + "; # from solver's max_iter parameter\n");
		dmlScript.append("num_iters_per_epoch = ceil(num_images/BATCH_SIZE);\n");
		 
		dmlScript.append("while(iter < maxiter) { \n");
		dmlScript.append("\t############################### BEGIN FEED FORWARD #############################\n" );
		TabbedStringBuilder tDmlScript = new TabbedStringBuilder(dmlScript, 1);
		for(Layer layer : dmlNet) {
			layer.generateForwardDML(tDmlScript); 
		}
		
		dmlScript.append("\n\t############################## END FEED FORWARD ###############################\n");
		
		dmlScript.append("\n\t############################## BEGIN BACK PROPAGATION #########################\n");
		
		for(int i = dmlNet.size()-1; i >= 0; i--) {
			Layer layer = dmlNet.get(i);
			layer.generateBackwardDML(tDmlScript); 
		}
		
		dmlScript.append("\n\t############################## END BACK PROPAGATION ###########################\n");
		
		dmlScript.append("\n\t############################## BEGIN UPDATE WEIGHTS AND BIAS #########################\n");
		setLocalLearningRate(dmlScript, solver);
		
		for(Layer layer : dmlNet) {
			if(layer.weightVar != null) {
				tDmlScript.append("# Update weights for the " + layer.param.getType() + " layer " + layer.param.getName() + "\n");
				
				String gradExpression = layer.gradientPrefix + layer.weightVar;
				
				if(!solver.hasRegularizationType() || 
						(solver.hasRegularizationType() && solver.getRegularizationType().compareToIgnoreCase("L2") == 0)) {
					if(!solver.hasWeightDecay())
						throw new DMLRuntimeException("weight_decay parameter required for L2 regularization");
					gradExpression += " - lambda * " + layer.weightVar;
					gradExpression = "(" + gradExpression + ")";
				}
				else {
					throw new DMLRuntimeException("Only L2 regularization is supported");
				}
				String updateExpression = "local_" + Barista.step_size + "*" + gradExpression;
				if(Barista.USE_MOMENTUM) {
					updateExpression = "(mu * " + layer.updatePrefix + layer.weightVar + ") + "
							+ updateExpression;
							// + "(1 - mu)*" + updateExpression;
				}
				tDmlScript.append(""+ layer.updatePrefix + layer.weightVar + " = " + updateExpression + ";\n");
				tDmlScript.append(""+ layer.weightVar + " = " + layer.weightVar + " + " +layer.updatePrefix + layer.weightVar + ";\n");
				
				tDmlScript.append("\n");
				tDmlScript.append("# Update bias for the " + layer.param.getType() + " layer " + layer.param.getName() + "\n");
				updateExpression = "(local_" + Barista.step_size + "*" + layer.gradientPrefix + layer.biasVar + ")";
				if(Barista.USE_MOMENTUM) {
					updateExpression = "(mu * " + layer.updatePrefix + layer.biasVar + ") + " 
							+ updateExpression;
							// + "(1 - mu)*" + updateExpression;
				}
				tDmlScript.append(layer.updatePrefix + layer.biasVar + " = " + updateExpression + ";\n");
				tDmlScript.append(layer.biasVar + " = " + layer.biasVar + " + " +layer.updatePrefix + layer.biasVar + ";\n");
			}
		}
		dmlScript.append("\n\t############################## END UPDATE WEIGHTS AND BIAS #########################\n");
		
		if(solver.hasDisplay()) {
			dmlScript.append("\n\tif(iter %% " + solver.getDisplay() + " == 0) {\n");
			dmlScript.append("\n\t\t############################## BEGIN VALIDATION #########################\n");
			appendTestValidationSet(dmlScript, dmlNet, solver, "X" + Barista.validationVarsuffix, 
					"y" + Barista.validationVarsuffix, Barista.numValidation, 2);
			dmlScript.append("\t\tprint(\"ITER=\" + "
					+ "iter + \" Validation set accuracy (%): \" + acc);\n");
			dmlScript.append("\n\t\t############################## END VALIDATION #########################\n");
			dmlScript.append("\n\t}\n");
		}
		
		dmlScript.append("\n\t" + "iter = iter + 1;\n");
		dmlScript.append("}\n");
		for(Layer layer : dmlNet) {
			String finalizeDML = layer.generateFinalizeDML(); 
			if(finalizeDML != null && !finalizeDML.trim().equals("")) {
				dmlScript.append(finalizeDML + "\n");
			}
		}
		
		dmlScript.append("\n############################## BEGIN TEST #########################\n");
		// TODO:
		dmlScript.append("Xt = read(\"mnist_test.X\")");
		dmlScript.append("yt = read(\"mnist_test.y\")");
		dmlScript.append(Barista.numTest + " = nrow(yt)");
		appendTestValidationSet(dmlScript, dmlNet, solver, "X" + Barista.testVarsuffix, 
				"y" + Barista.testVarsuffix, Barista.numTest, 0);
		dmlScript.append("print(\"Final accuracy on test set (%): \" + acc);\n");
		dmlScript.append("\n############################## END TEST #########################\n");
		
		return dmlScript.toString();
	}
	
	private void appendTestValidationSet(StringBuilder dmlScript, 
			ArrayList<Layer> dmlNet, SolverParameter solver,
			String X, String y, String numValues, int numTabs) throws DMLRuntimeException {
		String allTabs = "";
		for(int i = 0; i < numTabs; i++)
			allTabs += "\t";
		
		Layer firstLayer = dmlNet.get(0);
		String oldOutputVar = firstLayer.outputVar;
		String oldNumPoints = firstLayer.output_shape.get(0);
		
		if(!(firstLayer instanceof DataLayer)) {
			throw new DMLRuntimeException("Expected the first layer to be data layer for current validation implementation.");
		}
		else {
			firstLayer.outputVar = X;
			firstLayer.output_shape.set(0, numValues);
		}
		
		Layer lastLayer = dmlNet.get(dmlNet.size()-1);
		if(!(lastLayer instanceof SoftmaxWithLossLayer)) {
			throw new DMLRuntimeException("Expected the last layer to be SoftmaxWithLossLayer layer for current validation implementation.");
		}
		
		String lastOutputLayerVar = null;
		TabbedStringBuilder tDmlScript = new TabbedStringBuilder(dmlScript, numTabs);
		for(Layer layer : dmlNet) {
			if(layer != firstLayer && layer != lastLayer) {
				layer.updateOutputShape();
				layer.generateForwardDML(tDmlScript);
				lastOutputLayerVar = layer.outputVar;
			}
		}
		dmlScript.append("\n" + allTabs + "# Output accuracy on validation set\n");
		dmlScript.append(allTabs + "pred = rowIndexMax(" + lastOutputLayerVar + ");\n");
		// TODO: Make yv more generic 
		dmlScript.append(allTabs + "acc = sum((pred == (" + y + " + 1)))*100/" + numValues + ";\n");
		
		firstLayer.outputVar = oldOutputVar;
		firstLayer.output_shape.set(0, oldNumPoints);
		for(Layer layer : dmlNet) {
			layer.updateOutputShape();
		}
	}
	
	private void setLocalLearningRate(StringBuilder dmlScript, SolverParameter solver) throws DMLRuntimeException {
		dmlScript.append("\n\t# Set local learning rate using policy: " + solver.getLrPolicy());
		if(solver.getLrPolicy().equalsIgnoreCase("const")) {
			dmlScript.append("\n\tlocal_" + Barista.step_size + " = " + Barista.step_size + ";");
		}
		else if(solver.getLrPolicy().equalsIgnoreCase("exp")) {
			if(!solver.hasGamma())
				throw new DMLRuntimeException("gamma parameter required for lr_policy:" +  solver.getLrPolicy());
			dmlScript.append("\n\tlocal_" + Barista.step_size + " = " + Barista.step_size + " * " 
				+ "gamma^(iter+1);");
		}
		else if(solver.getLrPolicy().equalsIgnoreCase("step")) {
			if(!(solver.hasGamma() && solver.hasStepsize()))
				throw new DMLRuntimeException("gamma and step_size parameter required for lr_policy:" +  solver.getLrPolicy());
			dmlScript.append("\n\tlocal_" + Barista.step_size + " = " + Barista.step_size + " * gamma" + 
					"^((iter+1)/" + MathUtils.toDouble(solver.getStepsize()) + ");");
		}
		else if(solver.getLrPolicy().equalsIgnoreCase("inv")) {
			if(!(solver.hasGamma() && solver.hasPower()))
				throw new DMLRuntimeException("gamma and power parameter required for lr_policy:" +  solver.getLrPolicy());
			dmlScript.append("\n\tlocal_" + Barista.step_size + " = " + Barista.step_size + " * (1 + gamma" + 
					"*(iter+1))^" + MathUtils.toDouble(-solver.getPower()) + ";");
		}
		else {
			throw new DMLRuntimeException("Unsupported learning rate policy:" + solver.getLrPolicy() + ". "
					+ "The valid values are const, inv, ");
		}
		dmlScript.append("\n\tlocal_" + Barista.step_size + " = local_" + Barista.step_size + "/n" + ";\n"); 
	}
}
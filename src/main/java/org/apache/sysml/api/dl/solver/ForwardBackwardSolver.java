package org.apache.sysml.api.dl.solver;

import java.util.ArrayList;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.layer.Layer;
import org.apache.sysml.api.dl.utils.MathUtils;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.SolverParameter;

public class ForwardBackwardSolver {
	
	public String getForwardBackwardDML(ArrayList<Layer> dmlNet, SolverParameter solver) throws DMLRuntimeException {
		StringBuilder dmlScript = new StringBuilder();
		
		// TODO: Adding defaults for MNIST: Remove it later
		dmlScript.append(Barista.numClasses + " = ifdef($" + Barista.numClasses + ", 1);\n");
		dmlScript.append(Barista.inputHeight + " = ifdef($" + Barista.inputHeight + ", 28);\n");
		dmlScript.append(Barista.inputWidth + " = ifdef($" + Barista.inputWidth + ", 28);\n");
		dmlScript.append(Barista.numClasses + " = ifdef($" + Barista.numClasses + ", 10);\n");
		dmlScript.append(Barista.numValidation + " = ifdef($" + Barista.numValidation + ", 5000);\n");
		
		if(solver.hasMomentum() && solver.getMomentum() != 0) {
			Barista.useMomentum = true;
		}
		
		if(solver.hasWeightDecay()) {
			dmlScript.append("# Regularization constant\n");
			dmlScript.append("lambda = " +  MathUtils.toDouble(solver.getWeightDecay()) + ";\n");
		}
		if(Barista.useMomentum) {
			dmlScript.append("# Momentum\n");
			dmlScript.append("mu = " +  MathUtils.toDouble(solver.getMomentum()) + ";\n");
		}
		if(solver.hasGamma())
			dmlScript.append("gamma = " + MathUtils.toDouble(solver.getGamma()) + ";\n");
		else
			throw new DMLRuntimeException("Expected gamma parameter in the solver");
		
		for(Layer layer : dmlNet) {
			String setupDML = layer.getSetupDML(); 
			if(setupDML != null && !setupDML.trim().equals("")) {
				dmlScript.append("\n# Setup the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append(layer.getSetupDML());
			}
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
		for(Layer layer : dmlNet) {
			String forwardDML = layer.getForwardDML(); 
			if(forwardDML != null && !forwardDML.trim().equals("")) {
				dmlScript.append("\n\t# Perform forward pass on the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append("\t" + forwardDML + "\n");
			}
		}
		
		dmlScript.append("\n\t############################## END FEED FORWARD ###############################\n");
		dmlScript.append("\n\t############################## BEGIN BACK PROPAGATION #########################\n");
		
		for(int i = dmlNet.size()-1; i >= 0; i--) {
			Layer layer = dmlNet.get(i);
			String backwardDML = layer.getBackwardDML(); 
			if(backwardDML != null && !backwardDML.trim().equals("")) {
				dmlScript.append("\n\t# Perform backward pass on the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append("\t" + backwardDML + "\n");
			}
		}
		
		dmlScript.append("\n\t############################## END BACK PROPAGATION ###########################\n");
		
		dmlScript.append("\n\t############################## BEGIN UPDATE WEIGHTS AND BIAS #########################\n");
		setLocalLearningRate(dmlScript, solver);
		
		for(Layer layer : dmlNet) {
			if(layer.weightVar != null) {
				dmlScript.append("\n\t# Update weights for the " + layer.param.getType() + " layer " + layer.param.getName() + "\n");
				
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
				if(Barista.useMomentum) {
					updateExpression = "(mu * " + layer.updatePrefix + layer.weightVar + ") + " 
							+ "(1 - mu)*" + updateExpression;
				}
				dmlScript.append("\t" + layer.updatePrefix + layer.weightVar + " = " + updateExpression + ";\n");
				dmlScript.append("\t" + layer.weightVar + " = " + layer.weightVar + " + " +layer.updatePrefix + layer.weightVar + ";\n");
				
				dmlScript.append("\n\t# Update bias for the " + layer.param.getType() + " layer " + layer.param.getName() + "\n");
				updateExpression = "(local_" + Barista.step_size + "*" + layer.gradientPrefix + layer.biasVar + ")";
				if(Barista.useMomentum) {
					updateExpression = "(mu * " + layer.updatePrefix + layer.biasVar + ") + " 
							+ "(1 - mu)*" + updateExpression;
				}
				dmlScript.append("\t" + layer.updatePrefix + layer.biasVar + " = " + updateExpression + ";\n");
				dmlScript.append("\t" + layer.biasVar + " = " + layer.biasVar + " + " +layer.updatePrefix + layer.biasVar + ";\n");
			}
		}
		dmlScript.append("\n\t############################## END UPDATE WEIGHTS AND BIAS #########################\n");
		
		dmlScript.append("\n\t" + "iter = iter + 1;\n");
		dmlScript.append("}\n");
		for(Layer layer : dmlNet) {
			String finalizeDML = layer.getFinalizeDML(); 
			if(finalizeDML != null && !finalizeDML.trim().equals("")) {
				dmlScript.append(finalizeDML + "\n");
			}
		}
		return dmlScript.toString();
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
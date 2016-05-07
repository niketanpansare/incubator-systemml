package org.apache.sysml.api.dl.solver;

import java.util.ArrayList;

import org.apache.sysml.api.dl.Barista;
import org.apache.sysml.api.dl.layer.Layer;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.SolverParameter;

public class ForwardBackwardSolver {
	
	public String getForwardBackwardDML(ArrayList<Layer> dmlNet, SolverParameter solver) throws DMLRuntimeException {
		StringBuilder dmlScript = new StringBuilder();
		
		dmlScript.append("max_iterations = " + solver.getMaxIter() + "\n");
		// TODO: Adding defaults for MNIST: Remove it later
		dmlScript.append(Barista.numClasses + " = ifdef($" + Barista.numClasses + ", 1);\n");
		dmlScript.append(Barista.inputHeight + " = ifdef($" + Barista.inputHeight + ", 28);\n");
		dmlScript.append(Barista.inputWidth + " = ifdef($" + Barista.inputWidth + ", 28);\n");
		dmlScript.append(Barista.numClasses + " = ifdef($" + Barista.numClasses + ", 10);\n");
		dmlScript.append(Barista.numValidation + " = ifdef($" + Barista.numValidation + ", 5000);\n");
		
		for(Layer layer : dmlNet) {
			String setupDML = layer.getSetupDML(); 
			if(setupDML != null && !setupDML.trim().equals("")) {
				dmlScript.append("\n# Setup the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append(layer.getSetupDML());
			}
			layer.updateOutputShape();
		}
		dmlScript.append("iter = 0;\n");
		dmlScript.append("while(iter < max_iterations) { \n");
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
}
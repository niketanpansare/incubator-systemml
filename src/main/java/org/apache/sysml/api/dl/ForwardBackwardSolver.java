package org.apache.sysml.api.dl;

import java.util.ArrayList;

import org.apache.sysml.api.dl.layers.Layer;
import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.SolverParameter;

public class ForwardBackwardSolver {
	
	public String getForwardBackwardDML(ArrayList<Layer> dmlNet, SolverParameter solver) throws DMLRuntimeException {
		StringBuilder dmlScript = new StringBuilder();
		dmlScript.append("max_iterations = " + solver.getMaxIter() + "\n");
		for(Layer layer : dmlNet) {
			String setupDML = layer.getSetupDML(); 
			if(setupDML != null && !setupDML.trim().equals("")) {
				dmlScript.append("\n # Setup the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append(layer.getSetupDML() + "\n");
			}
			layer.updateOutputShape();
		}
		// dmlScript.append("iter = 1;\n");
		// dmlScript.append("while(iter < " + solver.getMaxIter() + ") {\n");
		dmlScript.append("for(iter in 1:max_iterations) { \n");
		for(Layer layer : dmlNet) {
			String forwardDML = layer.getForwardDML(); 
			if(forwardDML != null && !forwardDML.trim().equals("")) {
				dmlScript.append("\n\t # Perform forward pass on the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append("\t" + forwardDML + "\n");
			}
		}
		dmlScript.append("\t # TODO: Backward propogation\n");
		for(Layer layer : dmlNet) {
			String backwardDML = layer.getBackwardDML(); 
			if(backwardDML != null && !backwardDML.trim().equals("")) {
				dmlScript.append("\n\t # Perform backward pass on the " + layer.param.getType() + " layer " + layer.param.getName() + " \n");
				dmlScript.append("\t" + backwardDML + "\n");
			}
		}
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

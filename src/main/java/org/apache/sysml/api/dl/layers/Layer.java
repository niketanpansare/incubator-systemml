package org.apache.sysml.api.dl.layers;

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;

import caffe.Caffe.LayerParameter;

public abstract class Layer {
	public static int id = 1;
	public LayerParameter param;
	
	// Used by next layer for forward propogation
	public String outputVar;
	public ArrayList<String> output_shape = new ArrayList<String>();
	public ArrayList<Layer> bottom = new ArrayList<Layer>();
	
	// TODO:
	// Used by previous layer for backward propogation
	
	public Layer(LayerParameter param, String outputVar) {
		this.param = param;
		this.outputVar = outputVar;
	}
	
	public abstract String getSetupDML() throws DMLRuntimeException;
	public abstract String getForwardDML() throws DMLRuntimeException;
	public abstract String getBackwardDML() throws DMLRuntimeException;
	public abstract String getFinalizeDML() throws DMLRuntimeException;
	public abstract void updateOutputShape()  throws DMLRuntimeException;
	
	protected String getInputShape() {
		return "input_shape=[" + getBottomLayerOutputShape(0) + "," + getBottomLayerOutputShape(1) + "," + getBottomLayerOutputShape(2) + "," + getBottomLayerOutputShape(3) + "]";
	}
	
	protected void checkInput() throws DMLRuntimeException {
		if(bottom.size() == 0) {
			throw new DMLRuntimeException("The layer " + param.getName() + " cannot be the bottom-most layer");
		}
		else if(bottom.size() > 1) {
			throw new DMLRuntimeException("Multi-input " + param.getName() + " is not implemented");
		}
	}
	
	public String toString() {
		String ret = param.getName() + " <- [";
		for(int i = 0; i < param.getBottomCount(); i++) {
			 ret += " " + param.getBottom(i);
		}
		ret += "]";
		return ret;
	}
	
	public String getOutputShape(int index) {
		String ret = "??";
		try {
			ret = output_shape.get(index);
		} catch(Exception e) {}
		return ret;
	}
	
	public String getBottomLayerOutputShape(int index) {
		String ret = "??";
		try {
			ret = bottom.get(0).getOutputShape(index);
		} catch(Exception e) {}
		return ret;
	}
}

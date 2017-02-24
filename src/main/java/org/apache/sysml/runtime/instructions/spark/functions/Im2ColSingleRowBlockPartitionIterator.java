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


package org.apache.sysml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

import scala.Tuple2;

public class Im2ColSingleRowBlockPartitionIterator implements Iterator<Tuple2<MatrixIndexes, MatrixBlock>> {
	Iterator<Tuple2<MatrixIndexes, MatrixBlock>> blocks;
	MatrixCharacteristics mcRdd; MatrixCharacteristics mcOut;
	ConvolutionParameters params;
	boolean processingCurrentBlock = true;
	Tuple2<MatrixIndexes, MatrixBlock> currentBlock;
	
	public Im2ColSingleRowBlockPartitionIterator(Tuple2<MatrixIndexes, MatrixBlock> arg0, MatrixCharacteristics mcRdd, ConvolutionParameters params, MatrixCharacteristics mcOut) {
		this.mcRdd = mcRdd;
		this.params = params;
		this.mcOut = mcOut;
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> tmp = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		tmp.add(arg0);
		blocks = tmp.iterator();
	}
	public Im2ColSingleRowBlockPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0, MatrixCharacteristics mcRdd, ConvolutionParameters params, MatrixCharacteristics mcOut) {
		blocks = arg0;
		this.mcRdd = mcRdd;
		this.params = params;
		this.mcOut = mcOut;
	}

	@Override
	public boolean hasNext() {
		return blocks.hasNext() || hasNextN() || hasNextC();
	}
	
	long startingC = -1;
	long currentC = 0; long endC = -1;
	long currentN = 0; long endN = -1;
	private boolean hasNextN() {
		return currentN <= endN;
	}
	private boolean hasNextC() {
		return currentC <= endC;
	}
	private void updateCAndN() {
		if(hasNextC()) {
			currentC++;
		}
		else {
			currentN++;
			currentC = startingC;
		}
	}
	
	private void setCurrentBlock(Tuple2<MatrixIndexes, MatrixBlock> blk) {
		currentBlock = blk;
		currentN = (blk._1.getRowIndex() - 1)*mcRdd.getRowsPerBlock();
		endN = Math.min(blk._1.getRowIndex()*mcRdd.getRowsPerBlock()-1, mcRdd.getRows()-1);
		
		long [] ret = new long[3];
		
		long startColIndex = (blk._1.getColumnIndex() - 1)*mcRdd.getColsPerBlock();
		computeTensorIndexes(startColIndex, ret, params.H, params.W);
		startingC = ret[0];
		currentC = ret[0];
		
		long endColIndex = Math.min(blk._1.getColumnIndex()*mcRdd.getColsPerBlock()-1, mcRdd.getCols()-1);
		computeTensorIndexes(endColIndex, ret, params.H, params.W);
		endC = ret[0];
	}
	
	LinkedList<Tuple2<MatrixIndexes, MatrixBlock>> precomputedOutputs = new LinkedList<Tuple2<MatrixIndexes,MatrixBlock>>();

	@Override
	public Tuple2<MatrixIndexes, MatrixBlock> next() {
		if(precomputedOutputs.size() == 0) {
			if(!hasNextN() && !hasNextC()) {
				setCurrentBlock(blocks.next());
			}
			// Output at maximum RSPQ values at a time
			computeIm2ColOutputs();
			updateCAndN();
		}
		return precomputedOutputs.remove();
	}
	
	// Fills in precomputedOutputs data structure
	private void computeIm2ColOutputs() {
		HashMap<MatrixIndexes, MatrixBlock> tmp = new HashMap<MatrixIndexes, MatrixBlock>();
		
		MatrixIndexes mi = currentBlock._1;
		MatrixBlock mb = currentBlock._2;
		// TODO:
		
		for(Entry<MatrixIndexes, MatrixBlock> entry : tmp.entrySet()) {
			precomputedOutputs.add(new Tuple2<MatrixIndexes, MatrixBlock>(entry.getKey(), entry.getValue()));
		}
	}

	private static void computeTensorIndexes(long j, long [] ret, int H, int W) {
		ret[0] = j / (H*W);
		ret[1] = (j - ret[0]*(H*W))/W;
		ret[2] = j % W;
	}
}

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

package org.apache.sysml.test.integration.functions.paramserv;

import java.util.stream.IntStream;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.paramserv.dp.DataPartitionLocalScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.dp.LocalDataPartitioner;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.internal.ArrayComparisonFailure;

public abstract class BaseDataPartitionerTest {

	protected static final int ROW_SIZE = 4000;
	protected static final int COL_SIZE = 2000;
	protected static final int WORKER_NUM = 4;
	

	public static void assertArrayEquals(double[] expecteds,
			double[] actuals, double delta) throws ArrayComparisonFailure {
		// TODO: For now, we run MR tests under this test suite
		Assert.assertArrayEquals(expecteds, actuals, delta);
	}
	
	public static void assertEquals(int expected, int actual) {
		// TODO: For now, we run MR tests under this test suite
		Assert.assertEquals(expected, actual);
    }
	

	protected MatrixBlock[] generateData() {
		double[][] df = new double[BaseDataPartitionerTest.ROW_SIZE][BaseDataPartitionerTest.COL_SIZE];
		for (int i = 0; i < BaseDataPartitionerTest.ROW_SIZE; i++) {
			for (int j = 0; j < BaseDataPartitionerTest.COL_SIZE; j++) {
				df[i][j] = i * BaseDataPartitionerTest.COL_SIZE + j;
			}
		}
		double[] dl = new double[BaseDataPartitionerTest.ROW_SIZE];
		for (int i = 0; i < BaseDataPartitionerTest.ROW_SIZE; i++) {
			dl[i] = i;
		}
		MatrixBlock fmb = DataConverter.convertToMatrixBlock(df);
		MatrixBlock lmb = DataConverter.convertToMatrixBlock(dl, true);
		return new MatrixBlock[] { fmb, lmb };
	}

	protected double[] generateExpectedData(int from, int to) {
		return IntStream.range(from, to).mapToDouble(i -> (double) i).toArray();
	}

	protected DataPartitionLocalScheme.Result launchLocalDataPartitionerDC() {
		LocalDataPartitioner dp = new LocalDataPartitioner(Statement.PSScheme.DISJOINT_CONTIGUOUS);
		MatrixBlock[] mbs = generateData();
		return dp.doPartitioning(WORKER_NUM, mbs[0], mbs[1]);
	}

	protected DataPartitionLocalScheme.Result launchLocalDataPartitionerDR(MatrixBlock[] mbs) {
		ParamservUtils.SEED = System.nanoTime();
		LocalDataPartitioner dp = new LocalDataPartitioner(Statement.PSScheme.DISJOINT_RANDOM);
		return dp.doPartitioning(WORKER_NUM, mbs[0], mbs[1]);
	}

	protected DataPartitionLocalScheme.Result launchLocalDataPartitionerDRR() {
		LocalDataPartitioner dp = new LocalDataPartitioner(Statement.PSScheme.DISJOINT_ROUND_ROBIN);
		MatrixBlock[] mbs = generateData();
		return dp.doPartitioning(WORKER_NUM, mbs[0], mbs[1]);
	}

	protected DataPartitionLocalScheme.Result launchLocalDataPartitionerOR() {
		LocalDataPartitioner dp = new LocalDataPartitioner(Statement.PSScheme.OVERLAP_RESHUFFLE);
		MatrixBlock[] mbs = generateData();
		return dp.doPartitioning(WORKER_NUM, mbs[0], mbs[1]);
	}
}

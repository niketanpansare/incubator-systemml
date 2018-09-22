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

package org.apache.sysml.parser.dml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.ParameterExpression;
import org.apache.sysml.parser.common.CommonSyntacticValidator;
import org.apache.sysml.parser.common.CustomErrorListener;
import org.apache.sysml.parser.common.ExpressionInfo;
import org.apache.sysml.parser.dml.DmlParser.AccumulatorAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.AddSubExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.AssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.AtomicExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanAndExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanNotExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BooleanOrExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.BuiltinFunctionExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.CommandlineParamExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.CommandlinePositionExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstDoubleIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstFalseExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstIntIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstStringIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ConstTrueExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.DataIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.DataIdentifierContext;
import org.apache.sysml.parser.dml.DmlParser.ExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ExternalFunctionDefExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ForStatementContext;
import org.apache.sysml.parser.dml.DmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.FunctionCallMultiAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IfStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IfdefAssignmentStatementContext;
import org.apache.sysml.parser.dml.DmlParser.ImportStatementContext;
import org.apache.sysml.parser.dml.DmlParser.IndexedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.InternalFunctionDefExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.IterablePredicateColonExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.IterablePredicateSeqExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MatrixDataTypeCheckContext;
import org.apache.sysml.parser.dml.DmlParser.MatrixMulExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.Ml_typeContext;
import org.apache.sysml.parser.dml.DmlParser.ModIntDivExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MultDivExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.MultiIdExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ParForStatementContext;
import org.apache.sysml.parser.dml.DmlParser.ParameterizedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.PathStatementContext;
import org.apache.sysml.parser.dml.DmlParser.PowerExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ProgramrootContext;
import org.apache.sysml.parser.dml.DmlParser.RelationalExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.SimpleDataIdentifierExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.StrictParameterizedExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.StrictParameterizedKeyValueStringContext;
import org.apache.sysml.parser.dml.DmlParser.TypedArgAssignContext;
import org.apache.sysml.parser.dml.DmlParser.TypedArgNoAssignContext;
import org.apache.sysml.parser.dml.DmlParser.UnaryExpressionContext;
import org.apache.sysml.parser.dml.DmlParser.ValueTypeContext;
import org.apache.sysml.parser.dml.DmlParser.WhileStatementContext;


class InlineableFunction {
	// <input/output varname, ___input/output___>
	HashMap<String, String> inOutVarMapping;
	HashMap<String, String> internalVarMapping;
	StringBuilder body;
	String id;
	
	String getInlinedDML(ArrayList<String> inputs, ArrayList<String> outputs) {
		String _body = body.toString();
		for(Entry<String, String> kv : inOutVarMapping.entrySet()) {
			_body = _body.replaceAll(kv.getValue(), kv.getKey());
		}
		return _body;
	}
}

public class InlineDmlSyntacticValidator extends CommonSyntacticValidator implements DmlListener {
	
	public InlineDmlSyntacticValidator(CustomErrorListener errorListener, Map<String, String> argVals,
			String sourceNamespace, Set<String> prepFunctions) {
		super(errorListener, argVals, sourceNamespace, prepFunctions);
		
	}

	@Override
	protected ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName,
			ArrayList<ParameterExpression> paramExpression, Token fnName) {
		
		return null;
	}

	Stack<InlineableFunction> fnStack = new Stack<InlineableFunction>();
	HashMap<String, InlineableFunction> udfs;

	@Override
	public void exitAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.info.sb = new StringBuilder();
			ctx.targetList.dataInfo.appendText(ctx.info.sb, ctx.targetList);
			ctx.info.sb.append(ctx.op.getText());
			ctx.source.info.appendText(ctx.info.sb, ctx.source);
		}
	}

	@Override
	public void exitAddSubExpression(AddSubExpressionContext ctx) {
		binaryExpressionHelper(ctx.info, ctx.left, ctx.op, ctx.right);
	}

	@Override
	public void exitAssignmentStatement(AssignmentStatementContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.info.sb = new StringBuilder();
			ctx.targetList.dataInfo.appendText(ctx.info.sb, ctx.targetList);
			ctx.info.sb.append(ctx.op.getText());
			ctx.source.info.appendText(ctx.info.sb, ctx.source);
		}
	}
	
	private void binaryExpressionHelper(ExpressionInfo info, ExpressionContext left, org.antlr.v4.runtime.Token op, ExpressionContext right) {
		if(!fnStack.isEmpty()) {
			info.sb = new StringBuilder();
			left.info.appendText(info.sb, left);
			info.sb.append(op.getText());
			right.info.appendText(info.sb, right);
		}
	}

	@Override
	public void exitAtomicExpression(AtomicExpressionContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.info.sb = new StringBuilder();
			ctx.info.sb.append("(");
			ctx.left.info.appendText(ctx.info.sb, ctx.left);
			ctx.info.sb.append(")");
		}
	}

	@Override
	public void exitBooleanAndExpression(BooleanAndExpressionContext ctx) {
		binaryExpressionHelper(ctx.info, ctx.left, ctx.op, ctx.right);		
	}

	@Override
	public void exitBooleanNotExpression(BooleanNotExpressionContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.info.sb = new StringBuilder();
			ctx.info.sb.append("!");
			ctx.left.info.appendText(ctx.info.sb, ctx.left);
		}
	}

	@Override
	public void exitBooleanOrExpression(BooleanOrExpressionContext ctx) {
		binaryExpressionHelper(ctx.info, ctx.left, ctx.op, ctx.right);
	}

	@Override
	public void exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		InlineableFunction fn = getUDF(ctx.name.getText());
		if(fn != null) {
			// Skip the inlining of potentially simple UDFs are RHS is not available.
		}
	}
	
	private InlineableFunction getUDF(String fnName) {
		if(udfs.containsKey(fnName)) {
			return udfs.get(fnName);
		}
		else if(fnName.contains("::")) {
			String [] elems = fnName.split("::");
			fnName = elems[0] + "::" + elems[1];
			if(elems.length == 2 && udfs.containsKey(fnName)) {
				return udfs.get(fnName);
			}
		}
		return null;
	}

	@Override
	public void exitCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		
		
	}

	@Override
	public void exitCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstFalseExpression(ConstFalseExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitConstTrueExpression(ConstTrueExpressionContext ctx) {
		
		
	}
	
	private void handleVariable(ExpressionContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.info.sb = new StringBuilder();
			String oldVarName = ctx.getText();
			if(!fnStack.peek().internalVarMapping.containsKey(oldVarName)) {
				String newVarName = "internal_fn_" + fnStack.peek().id + oldVarName;
				fnStack.peek().internalVarMapping.put(oldVarName, newVarName);
			}
			ctx.info.sb.append(fnStack.peek().internalVarMapping.get(oldVarName));
		}
	}
	
	private void handleVariable(DataIdentifierContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.dataInfo.sb = new StringBuilder();
			String oldVarName = ctx.getText();
			if(!fnStack.peek().internalVarMapping.containsKey(oldVarName)) {
				String newVarName = "internal_fn_" + fnStack.peek().id + oldVarName;
				fnStack.peek().internalVarMapping.put(oldVarName, newVarName);
			}
			ctx.dataInfo.sb.append(fnStack.peek().internalVarMapping.get(oldVarName));
		}
	}
	
	

	@Override
	public void exitDataIdExpression(DataIdExpressionContext ctx) {
		handleVariable(ctx);
	}

	@Override
	public void exitEveryRule(ParserRuleContext arg0) {
		
		
	}

	@Override
	public void exitExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		
		
	}

	@Override
	public void exitForStatement(ForStatementContext ctx) {
		// 'for' '(' iterVar=ID 'in' iterPred=iterablePredicate (',' parForParams+=strictParameterizedExpression)* ')' 
		// (body+=statement ';'* | '{' (body+=statement ';'* )*  '}')  # ForStatement
		if(!fnStack.isEmpty()) {
			
		}
		
	}

	@Override
	public void exitFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		InlineableFunction fn = getUDF(ctx.name.getText());
		if(fn != null) {
			// Perform inlining
			if(fnStack.pop() != fn) {
				throw new RuntimeException("Internal Error while inlining.");
			}
		}
	}

	@Override
	public void exitFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void exitIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void exitIfStatement(IfStatementContext ctx) {
		
		
	}

	@Override
	public void exitImportStatement(ImportStatementContext ctx) {
		
		
	}

	@Override
	public void exitIndexedExpression(IndexedExpressionContext ctx) {
		if(!fnStack.isEmpty()) {
			ctx.dataInfo.sb = new StringBuilder();
			String oldVarName = ctx.name.getText();
			if(!fnStack.peek().internalVarMapping.containsKey(oldVarName)) {
				String newVarName = "internal_fn_" + fnStack.peek().id + oldVarName;
				fnStack.peek().internalVarMapping.put(oldVarName, newVarName);
			}
			ctx.dataInfo.sb.append(fnStack.peek().internalVarMapping.get(oldVarName));
			ctx.dataInfo.sb.append("[");
			if(ctx.rowLower != null)
				ctx.rowLower.info.appendText(ctx.dataInfo.sb, ctx.rowLower);
			if(ctx.rowUpper != null) {
				ctx.dataInfo.sb.append(":");
				ctx.rowUpper.info.appendText(ctx.dataInfo.sb, ctx.rowUpper);
			}
			if(ctx.getText().contains(",")) {
				ctx.dataInfo.sb.append(",");
			}
			if(ctx.colLower != null)
				ctx.colLower.info.appendText(ctx.dataInfo.sb, ctx.colLower);
			if(ctx.rowUpper != null) {
				ctx.dataInfo.sb.append(":");
				ctx.colUpper.info.appendText(ctx.dataInfo.sb, ctx.colUpper);
			}
		}
		
	}

	@Override
	public void exitInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		
		
	}

	@Override
	public void exitIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		
		
	}

	@Override
	public void exitIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		
		
	}

	@Override
	public void exitMatrixMulExpression(MatrixMulExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMl_type(Ml_typeContext ctx) {
		
		
	}

	@Override
	public void exitModIntDivExpression(ModIntDivExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMultDivExpression(MultDivExpressionContext ctx) {
		
		
	}

	@Override
	public void exitMultiIdExpression(MultiIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitParameterizedExpression(ParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void exitParForStatement(ParForStatementContext ctx) {
		
		
	}

	@Override
	public void exitPathStatement(PathStatementContext ctx) {
		
		
	}

	@Override
	public void exitPowerExpression(PowerExpressionContext ctx) {
		
		
	}

	@Override
	public void exitProgramroot(ProgramrootContext ctx) {
		
		
	}

	@Override
	public void exitRelationalExpression(RelationalExpressionContext ctx) {
		
		
	}

	@Override
	public void exitSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		handleVariable(ctx);
		
	}

	@Override
	public void exitStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void exitStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {
		
		
	}

	@Override
	public void exitTypedArgAssign(TypedArgAssignContext ctx) {
		
		
	}

	@Override
	public void exitTypedArgNoAssign(TypedArgNoAssignContext ctx) {
		
		
	}

	@Override
	public void exitUnaryExpression(UnaryExpressionContext ctx) {
		
		
	}

	@Override
	public void exitValueType(ValueTypeContext ctx) {
		
		
	}

	@Override
	public void exitWhileStatement(WhileStatementContext ctx) {
		
		
	}

	@Override
	public String falseStringLiteral() {
		
		return null;
	}

	@Override
	protected Expression handleLanguageSpecificFunction(ParserRuleContext ctx, String functionName,
			ArrayList<ParameterExpression> paramExpressions) {
		
		return null;
	}

	@Override
	public String namespaceResolutionOp() {
		
		return null;
	}

	@Override
	public String trueStringLiteral() {
		
		return null;
	}

	@Override
	public void visitErrorNode(ErrorNode arg0) {
		
		
	}

	@Override
	public void visitTerminal(TerminalNode arg0) {
		
		
	}
	
	@Override
	public void enterAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterAddSubExpression(AddSubExpressionContext ctx) {
		
		
	}

	@Override
	public void enterAssignmentStatement(AssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterAtomicExpression(AtomicExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanAndExpression(BooleanAndExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanNotExpression(BooleanNotExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBooleanOrExpression(BooleanOrExpressionContext ctx) {
		
		
	}

	@Override
	public void enterBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		
		
	}

	@Override
	public void enterCommandlineParamExpression(CommandlineParamExpressionContext ctx) {
		
		
	}

	@Override
	public void enterCommandlinePositionExpression(CommandlinePositionExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstDoubleIdExpression(ConstDoubleIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstFalseExpression(ConstFalseExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstIntIdExpression(ConstIntIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstStringIdExpression(ConstStringIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterConstTrueExpression(ConstTrueExpressionContext ctx) {
		
		
	}

	@Override
	public void enterDataIdExpression(DataIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterEveryRule(ParserRuleContext arg0) {
	}

	@Override
	public void enterExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		
		
	}

	@Override
	public void enterForStatement(ForStatementContext ctx) {
		
		
	}

	@Override
	public void enterFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		InlineableFunction fn = getUDF(ctx.name.getText());
		if(fn != null) {
			fnStack.push(fn);
		}
	}

	@Override
	public void enterFunctionCallMultiAssignmentStatement(FunctionCallMultiAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterIfdefAssignmentStatement(IfdefAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void enterIfStatement(IfStatementContext ctx) {
		
		
	}

	@Override
	public void enterImportStatement(ImportStatementContext ctx) {
		
		
	}

	@Override
	public void enterIndexedExpression(IndexedExpressionContext ctx) {
		
		
	}

	@Override
	public void enterInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		
		
	}

	@Override
	public void enterIterablePredicateColonExpression(IterablePredicateColonExpressionContext ctx) {
		
		
	}

	@Override
	public void enterIterablePredicateSeqExpression(IterablePredicateSeqExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMatrixDataTypeCheck(MatrixDataTypeCheckContext ctx) {
		
		
	}

	@Override
	public void enterMatrixMulExpression(MatrixMulExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMl_type(Ml_typeContext ctx) {
		
		
	}

	@Override
	public void enterModIntDivExpression(ModIntDivExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMultDivExpression(MultDivExpressionContext ctx) {
		
		
	}

	@Override
	public void enterMultiIdExpression(MultiIdExpressionContext ctx) {
		
		
	}

	@Override
	public void enterParameterizedExpression(ParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void enterParForStatement(ParForStatementContext ctx) {
		
		
	}

	@Override
	public void enterPathStatement(PathStatementContext ctx) {
		
		
	}

	@Override
	public void enterPowerExpression(PowerExpressionContext ctx) {
		
		
	}

	@Override
	public void enterProgramroot(ProgramrootContext ctx) {
		
		
	}

	@Override
	public void enterRelationalExpression(RelationalExpressionContext ctx) {
		
		
	}

	@Override
	public void enterSimpleDataIdentifierExpression(SimpleDataIdentifierExpressionContext ctx) {
		
		
	}

	@Override
	public void enterStrictParameterizedExpression(StrictParameterizedExpressionContext ctx) {
		
		
	}

	@Override
	public void enterStrictParameterizedKeyValueString(StrictParameterizedKeyValueStringContext ctx) {
		
		
	}

	@Override
	public void enterTypedArgAssign(TypedArgAssignContext ctx) {
		
		
	}

	@Override
	public void enterTypedArgNoAssign(TypedArgNoAssignContext ctx) {
		
		
	}

	@Override
	public void enterUnaryExpression(UnaryExpressionContext ctx) {
		
		
	}

	@Override
	public void enterValueType(ValueTypeContext ctx) {
		
		
	}

	@Override
	public void enterWhileStatement(WhileStatementContext ctx) {
		
		
	}

}

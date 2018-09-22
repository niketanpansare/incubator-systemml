package org.apache.sysml.parser.dml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.ParameterExpression;
import org.apache.sysml.parser.common.CommonSyntacticValidator;
import org.apache.sysml.parser.common.CustomErrorListener;
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

public class InlineHelper extends CommonSyntacticValidator implements DmlListener {

	public InlineHelper(CustomErrorListener errorListener, Map<String, String> argVals, String sourceNamespace,
			Set<String> prepFunctions) {
		super(errorListener, argVals, sourceNamespace, prepFunctions);
		
	}

	@Override
	protected ConvertedDMLSyntax convertToDMLSyntax(ParserRuleContext ctx, String namespace, String functionName,
			ArrayList<ParameterExpression> paramExpression, Token fnName) {
		
		return null;
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
	
	String currentFunction;
	StringBuilder body;
	HashMap<String, String> inlineMap = new HashMap<>();

	@Override
	public void enterInternalFunctionDefExpression(InternalFunctionDefExpressionContext ctx) {
		currentFunction = ctx.name.getText();
		body = new StringBuilder();
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

	@Override
	public void exitAccumulatorAssignmentStatement(AccumulatorAssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void exitAddSubExpression(AddSubExpressionContext ctx) {
		
		
	}

	@Override
	public void exitAssignmentStatement(AssignmentStatementContext ctx) {
		
		
	}

	@Override
	public void exitAtomicExpression(AtomicExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanAndExpression(BooleanAndExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanNotExpression(BooleanNotExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBooleanOrExpression(BooleanOrExpressionContext ctx) {
		
		
	}

	@Override
	public void exitBuiltinFunctionExpression(BuiltinFunctionExpressionContext ctx) {
		
		
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

	@Override
	public void exitDataIdExpression(DataIdExpressionContext ctx) {
		
		
	}

	@Override
	public void exitEveryRule(ParserRuleContext arg0) {
		
		
	}

	@Override
	public void exitExternalFunctionDefExpression(ExternalFunctionDefExpressionContext ctx) {
		
		
	}

	@Override
	public void exitForStatement(ForStatementContext ctx) {
		
		
	}

	@Override
	public void exitFunctionCallAssignmentStatement(FunctionCallAssignmentStatementContext ctx) {
		
		
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

}

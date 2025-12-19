include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def R_Dialect : Dialect {
  let name = "r";
  let cppNamespace = "::r";
}

class R_Op<string mnemonic, list<Trait> traits = []> : Op<R_Dialect, mnemonic, traits>;

def R_ConstantOp : R_Op<"constant", [NoMemoryEffect, ConstantLike, AllTypesMatch<["value", "result"]>]> {
  let summary = "R constant";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def R_AddOp : R_Op<"add", [NoMemoryEffect, SameOperandsAndResultType]> {
  let summary = "R add";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def R_SubOp : R_Op<"sub", [NoMemoryEffect, SameOperandsAndResultType]> {
  let summary = "R sub";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def R_MulOp : R_Op<"mul", [NoMemoryEffect, SameOperandsAndResultType]> {
  let summary = "R mul";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def R_DivOp : R_Op<"div", [NoMemoryEffect, SameOperandsAndResultType]> {
  let summary = "R div";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def R_LoadOp : R_Op<"load", [
    NoMemoryEffect,
    TypesMatchWith<"result type matches element type of 'memref'",
                   "memref", "result",
                   "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "R load";
  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$memref `[` $indices `]` attr-dict `:` type($memref)";
}

def R_SliceOp : R_Op<"slice", [NoMemoryEffect, AttrSizedOperandSegments]> {
  let summary = "R slice";
  let arguments = (ins AnyMemRef:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       OptionalAttr<DenseBoolArrayAttr>:$drop_dims);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$source `[` $offsets `]` `sizes` `[` $sizes `]` `strides` `[` $strides `]` attr-dict `:` type($source) `->` type($result)";
}

def R_StoreOp : R_Op<"store", [
    TypesMatchWith<"type of 'value' matches element type of 'memref'",
                   "memref", "value",
                   "::llvm::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "R store";
  let arguments = (ins AnyType:$value, AnyMemRef:$memref, Variadic<Index>:$indices);
  let assemblyFormat = "$value `,` $memref `[` $indices `]` attr-dict `:` type($memref)";
}

def R_ForOp : R_Op<"for", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "R for";
  let arguments = (ins Index:$lower, Index:$upper, Index:$step);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$lower `to` $upper `step` $step attr-dict-with-keyword $body";
}

def R_ParallelForOp : R_Op<"parallel_for", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "R parallel for";
  let arguments = (ins Index:$lower, Index:$upper, Index:$step);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$lower `to` $upper `step` $step attr-dict-with-keyword $body";
}

def R_MatmulOp : R_Op<"matmul", [NoMemoryEffect]> {
  let summary = "R matmul";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def R_YieldOp : R_Op<"yield", [Terminator]> {
  let summary = "R yield";
  let assemblyFormat = "attr-dict";
}

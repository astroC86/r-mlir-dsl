#include "rdslmlir/Runtime/RuntimeAstLowerer.h"

#include <cmath>
#include <optional>

using namespace mlir;

namespace rdslmlir {

namespace {

std::optional<double> getNumericLiteral(const llvm::json::Object &obj) {
  auto kind = obj.getString("type");
  if (!kind || *kind != "Number") return std::nullopt;
  auto *val = obj.get("value");
  if (!val) return std::nullopt;
  if (auto num = val->getAsNumber()) return *num;
  if (auto numInt = val->getAsInteger()) return static_cast<double>(*numInt);
  return std::nullopt;
}

bool isFloatOnlyMath(llvm::StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Cases({"sqrt", "exp", "log", "log10", "log2"}, true)
      .Cases({"sin", "cos", "tan", "asin", "acos", "atan"}, true)
      .Cases({"sinh", "cosh", "tanh", "asinh", "acosh", "atanh"}, true)
      .Cases({"round", "floor", "ceiling", "trunc", "signif"}, true)
      .Default(false);
}

bool isNumericMath(llvm::StringRef name) {
  return name == "abs" || name == "sign";
}

bool isUnaryMath(llvm::StringRef name) {
  return isFloatOnlyMath(name) || isNumericMath(name);
}

} // namespace

const llvm::StringMap<AstLowerer::ExprHandler> &AstLowerer::getExprHandlers() {
  static const llvm::StringMap<ExprHandler> handlers = [] {
    llvm::StringMap<ExprHandler> map;
    map.try_emplace("Number", &AstLowerer::parseNumberExpr);
    map.try_emplace("Bool", &AstLowerer::parseBoolExpr);
    map.try_emplace("Var", &AstLowerer::parseVarExpr);
    map.try_emplace("Unary", &AstLowerer::parseUnaryExpr);
    map.try_emplace("Binary", &AstLowerer::parseBinaryExpr);
    map.try_emplace("AllocTensor", &AstLowerer::parseAllocTensorExpr);
    map.try_emplace("Clone", &AstLowerer::parseCloneExpr);
    map.try_emplace("ToTensor", &AstLowerer::parseToTensorExpr);
    map.try_emplace("ToBuffer", &AstLowerer::parseToBufferExpr);
    map.try_emplace("MaterializeInDestination",
                    &AstLowerer::parseMaterializeInDestinationExpr);
    map.try_emplace("Dim", &AstLowerer::parseDimExpr);
    map.try_emplace("Index", &AstLowerer::parseIndexExpr);
    map.try_emplace("Slice", &AstLowerer::parseSliceExpr);
    map.try_emplace("MatMul", &AstLowerer::parseMatMulExpr);
    map.try_emplace("Call", &AstLowerer::parseCallExpr);
    return map;
  }();
  return handlers;
}

Value AstLowerer::parseExpr(const llvm::json::Object &obj) {
  auto kind = obj.getString("type");
  if (!kind) return Value();
  auto &handlers = getExprHandlers();
  auto it = handlers.find(*kind);
  if (it == handlers.end()) return Value();
  return (this->*(it->second))(obj);
}

Value AstLowerer::parseNumberExpr(const llvm::json::Object &obj) {
  Type dtype = builder.getF64Type();
  if (auto *dtypeVal = obj.get("dtype")) dtype = parseType(*dtypeVal);
  auto val = obj.get("value");
  if (!val) return Value();
  if (llvm::isa<FloatType>(dtype)) {
    auto num = val->getAsNumber().value_or(0.0);
    return r::ConstantOp::create(builder, builder.getUnknownLoc(), dtype,
                                 builder.getFloatAttr(dtype, num));
  }
  if (llvm::isa<IndexType>(dtype)) {
    auto num = val->getAsInteger().value_or(0);
    return r::ConstantOp::create(builder, builder.getUnknownLoc(), dtype,
                                 builder.getIndexAttr(num));
  }
  auto num = val->getAsInteger().value_or(0);
  return r::ConstantOp::create(builder, builder.getUnknownLoc(), dtype,
                               builder.getIntegerAttr(dtype, num));
}

Value AstLowerer::parseBoolExpr(const llvm::json::Object &obj) {
  Type dtype = builder.getI1Type();
  if (auto *dtypeVal = obj.get("dtype")) dtype = parseType(*dtypeVal);
  auto val = obj.getBoolean("value");
  if (!val) return Value();
  return r::ConstantOp::create(builder, builder.getUnknownLoc(), dtype,
                               builder.getBoolAttr(*val));
}

Value AstLowerer::parseVarExpr(const llvm::json::Object &obj) {
  auto name = obj.getString("name");
  if (!name) return Value();
  return lookup(*name);
}

Value AstLowerer::parseUnaryExpr(const llvm::json::Object &obj) {
  auto op = obj.getString("op");
  auto *valObj = obj.getObject("value");
  if (!op || !valObj) return Value();
  Value inner = parseExpr(*valObj);
  if (*op == "+") return inner;
  if (*op == "-") {
    auto type = inner.getType();
    if (llvm::isa<FloatType>(type)) {
      auto zero = r::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getFloatAttr(type, 0.0));
      return r::SubOp::create(builder, builder.getUnknownLoc(), zero, inner);
    }
    auto zero = r::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                      builder.getIntegerAttr(type, 0));
    return r::SubOp::create(builder, builder.getUnknownLoc(), zero, inner);
  }
  return Value();
}

Value AstLowerer::parseBinaryExpr(const llvm::json::Object &obj) {
  auto op = obj.getString("op");
  auto *lhsObj = obj.getObject("lhs");
  auto *rhsObj = obj.getObject("rhs");
  if (!op || !lhsObj || !rhsObj) return Value();
  Value lhs = parseExpr(*lhsObj);
  Value rhs = parseExpr(*rhsObj);
  if (*op == "+") return r::AddOp::create(builder, builder.getUnknownLoc(), lhs, rhs);
  if (*op == "-") return r::SubOp::create(builder, builder.getUnknownLoc(), lhs, rhs);
  if (*op == "*") return r::MulOp::create(builder, builder.getUnknownLoc(), lhs, rhs);
  if (*op == "/") return r::DivOp::create(builder, builder.getUnknownLoc(), lhs, rhs);
  return Value();
}

Value AstLowerer::parseAllocTensorExpr(const llvm::json::Object &obj) {
  auto *typeVal = obj.get("tensorType");
  if (!typeVal) return Value();
  Type tensorType = parseType(*typeVal);
  if (!llvm::isa<RankedTensorType>(tensorType)) return Value();
  SmallVector<Value, 4> sizes;
  if (auto *sizesArr = obj.getArray("sizes")) {
    for (auto &szVal : *sizesArr) {
      auto *szObj = szVal.getAsObject();
      if (!szObj) return Value();
      sizes.push_back(ensureIndex(parseExpr(*szObj)));
    }
  }
  return bufferization::AllocTensorOp::create(builder, builder.getUnknownLoc(), tensorType, sizes);
}

Value AstLowerer::parseCloneExpr(const llvm::json::Object &obj) {
  auto *sourceObj = obj.getObject("source");
  if (!sourceObj) return Value();
  Value source = parseExpr(*sourceObj);
  return bufferization::CloneOp::create(builder, builder.getUnknownLoc(), source);
}

Value AstLowerer::parseToTensorExpr(const llvm::json::Object &obj) {
  auto *sourceObj = obj.getObject("source");
  auto *typeVal = obj.get("tensorType");
  if (!sourceObj || !typeVal) return Value();
  Value source = parseExpr(*sourceObj);
  Type tensorType = parseType(*typeVal);
  if (!llvm::isa<RankedTensorType>(tensorType)) return Value();
  return bufferization::ToTensorOp::create(builder, builder.getUnknownLoc(), tensorType, source);
}

Value AstLowerer::parseToBufferExpr(const llvm::json::Object &obj) {
  auto *sourceObj = obj.getObject("source");
  auto *typeVal = obj.get("bufferType");
  if (!sourceObj || !typeVal) return Value();
  Value source = parseExpr(*sourceObj);
  Type bufferType = parseType(*typeVal);
  if (!llvm::isa<MemRefType>(bufferType)) return Value();
  return bufferization::ToBufferOp::create(builder, builder.getUnknownLoc(), bufferType, source);
}

Value AstLowerer::parseMaterializeInDestinationExpr(const llvm::json::Object &obj) {
  auto *sourceObj = obj.getObject("source");
  auto *destObj = obj.getObject("dest");
  if (!sourceObj || !destObj) return Value();
  Value source = parseExpr(*sourceObj);
  Value dest = parseExpr(*destObj);
  auto loc = builder.getUnknownLoc();
  if (llvm::isa<BaseMemRefType>(dest.getType())) {
    bufferization::MaterializeInDestinationOp::create(
        builder, loc, Type(), source, dest, /*restrict=*/false, /*writable=*/true);
    return dest;
  }
  auto op = bufferization::MaterializeInDestinationOp::create(builder, loc, source, dest);
  if (op.getNumResults() == 0) return dest;
  return op.getResult();
}

Value AstLowerer::parseDimExpr(const llvm::json::Object &obj) {
  auto *targetObj = obj.getObject("target");
  auto axis = obj.getInteger("axis");
  if (!targetObj || !axis) return Value();
  Value target = parseExpr(*targetObj);
  if (!llvm::isa<MemRefType>(target.getType())) return Value();
  return memref::DimOp::create(builder, builder.getUnknownLoc(), target, *axis);
}

Value AstLowerer::parseIndexExpr(const llvm::json::Object &obj) {
  auto *targetObj = obj.getObject("target");
  if (!targetObj) return Value();
  Value target = parseExpr(*targetObj);
  auto memrefType = llvm::dyn_cast<MemRefType>(target.getType());
  if (!memrefType) return Value();
  SmallVector<Value, 4> indices;
  if (auto *idxArr = obj.getArray("indices")) {
    for (auto &idxVal : *idxArr) {
      auto *idxObj = idxVal.getAsObject();
      if (!idxObj) return Value();
      indices.push_back(ensureIndex(parseExpr(*idxObj)));
    }
  }
  return r::LoadOp::create(builder, builder.getUnknownLoc(), memrefType.getElementType(),
                           target, indices);
}

Value AstLowerer::parseSliceExpr(const llvm::json::Object &obj) {
  auto *sourceObj = obj.getObject("source");
  if (!sourceObj) return Value();
  Value source = parseExpr(*sourceObj);
  auto memrefType = llvm::dyn_cast<MemRefType>(source.getType());
  if (!memrefType) return Value();

  SmallVector<Value, 4> offsets;
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;

  auto *offsetsArr = obj.getArray("offsets");
  auto *sizesArr = obj.getArray("sizes");
  auto *stridesArr = obj.getArray("strides");
  if (!offsetsArr || !sizesArr) return Value();

  for (auto &offVal : *offsetsArr) {
    auto *offObj = offVal.getAsObject();
    if (!offObj) return Value();
    offsets.push_back(ensureIndex(parseExpr(*offObj)));
  }
  for (auto &sizeVal : *sizesArr) {
    auto *sizeObj = sizeVal.getAsObject();
    if (!sizeObj) return Value();
    sizes.push_back(ensureIndex(parseExpr(*sizeObj)));
  }

  if (stridesArr) {
    for (auto &strideVal : *stridesArr) {
      auto *strideObj = strideVal.getAsObject();
      if (!strideObj) return Value();
      strides.push_back(ensureIndex(parseExpr(*strideObj)));
    }
  } else {
    auto one = r::ConstantOp::create(builder, builder.getUnknownLoc(), builder.getIndexType(),
                                     builder.getIndexAttr(1));
    strides.assign(offsets.size(), one);
  }

  int64_t rank = memrefType.getRank();
  if ((int64_t)offsets.size() != rank || (int64_t)sizes.size() != rank ||
      (int64_t)strides.size() != rank) {
    return Value();
  }

  auto *dropArr = obj.getArray("dropDims");
  SmallVector<bool, 4> dropDims(rank, false);
  if (dropArr) {
    if ((int64_t)dropArr->size() != rank) return Value();
    for (int64_t i = 0; i < rank; ++i) {
      auto dropVal = (*dropArr)[i].getAsBoolean();
      if (!dropVal) return Value();
      dropDims[i] = *dropVal;
    }
  }

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (dropDims[i]) continue;
    int64_t dim = ShapedType::kDynamic;
    auto &sizeVal = (*sizesArr)[i];
    if (auto *sizeObj = sizeVal.getAsObject()) {
      if (auto k = sizeObj->getString("type")) {
        if (*k == "Number") {
          if (auto sizeInt = sizeObj->getInteger("value")) {
            dim = *sizeInt;
          } else if (auto sizeNum = sizeObj->getNumber("value")) {
            dim = static_cast<int64_t>(*sizeNum);
          }
        }
      }
    }
    shape.push_back(dim);
  }
  if (shape.empty()) return Value();

  auto parseStaticSize = [](const llvm::json::Value &val) -> int64_t {
    auto *obj = val.getAsObject();
    if (!obj) return ShapedType::kDynamic;
    auto kind = obj->getString("type");
    if (!kind || *kind != "Number") return ShapedType::kDynamic;
    if (auto intVal = obj->getInteger("value")) return *intVal;
    if (auto numVal = obj->getNumber("value")) return static_cast<int64_t>(*numVal);
    return ShapedType::kDynamic;
  };

  SmallVector<int64_t, 4> staticOffsets(rank, ShapedType::kDynamic);
  SmallVector<int64_t, 4> staticSizes(rank, ShapedType::kDynamic);
  SmallVector<int64_t, 4> staticStrides(rank, ShapedType::kDynamic);
  for (int64_t i = 0; i < rank; ++i) {
    staticSizes[i] = parseStaticSize((*sizesArr)[i]);
    if (dropDims[i]) staticSizes[i] = 1;
  }

  auto resultType = memref::SubViewOp::inferRankReducedResultType(
      shape, memrefType, staticOffsets, staticSizes, staticStrides);
  DenseBoolArrayAttr dropAttr;
  if (dropArr) dropAttr = builder.getDenseBoolArrayAttr(dropDims);
  return r::SliceOp::create(builder, builder.getUnknownLoc(), resultType, source, offsets, sizes,
                            strides, dropAttr);
}

Value AstLowerer::parseMatMulExpr(const llvm::json::Object &obj) {
  auto *lhsObj = obj.getObject("lhs");
  auto *rhsObj = obj.getObject("rhs");
  if (!lhsObj || !rhsObj) return Value();
  Value lhs = parseExpr(*lhsObj);
  Value rhs = parseExpr(*rhsObj);
  auto lhsType = llvm::dyn_cast<MemRefType>(lhs.getType());
  auto rhsType = llvm::dyn_cast<MemRefType>(rhs.getType());
  if (!lhsType || !rhsType) return Value();
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) return Value();
  if (lhsType.getElementType() != rhsType.getElementType()) return Value();

  if (!lhsType.isDynamicDim(1) && !rhsType.isDynamicDim(0) &&
      lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return Value();
  }

  SmallVector<int64_t, 2> shape;
  shape.push_back(lhsType.isDynamicDim(0) ? ShapedType::kDynamic : lhsType.getDimSize(0));
  shape.push_back(rhsType.isDynamicDim(1) ? ShapedType::kDynamic : rhsType.getDimSize(1));
  auto resultType = MemRefType::get(shape, lhsType.getElementType());
  return r::MatmulOp::create(builder, builder.getUnknownLoc(), resultType, lhs, rhs);
}

Value AstLowerer::parseCallExpr(const llvm::json::Object &obj) {
  auto name = obj.getString("name");
  if (!name) return Value();
  SmallVector<const llvm::json::Object *, 4> argObjs;
  if (auto *argsArr = obj.getArray("args")) {
    for (auto &argVal : *argsArr) {
      auto *argObj = argVal.getAsObject();
      if (!argObj) {
        (void)emitError("call argument is not an object");
        return Value();
      }
      argObjs.push_back(argObj);
    }
  }

  auto loc = builder.getUnknownLoc();
  llvm::StringRef nameRef = *name;

  if (isUnaryMath(nameRef)) {
    if (argObjs.empty()) {
      (void)emitError("math call missing operand");
      return Value();
    }
    if (argObjs.size() > 2) {
      (void)emitError("math call has too many arguments");
      return Value();
    }
    if (argObjs.size() == 2 &&
        nameRef != "log" && nameRef != "round" && nameRef != "signif") {
      (void)emitError("math call does not accept a second argument");
      return Value();
    }
    Value input = parseExpr(*argObjs[0]);
    if (!input) return Value();
    Type inputType = input.getType();
    Type elemType = inputType;
    if (auto memrefType = llvm::dyn_cast<MemRefType>(inputType)) {
      elemType = memrefType.getElementType();
    }
    if (isFloatOnlyMath(nameRef)) {
      if (!llvm::isa<FloatType>(elemType)) {
        (void)emitError("math call expects floating-point operand");
        return Value();
      }
    } else if (isNumericMath(nameRef)) {
      if (!llvm::isa<FloatType>(elemType) && !llvm::isa<IntegerType>(elemType)) {
        (void)emitError("math call expects numeric operand");
        return Value();
      }
    }

    FloatAttr baseAttr;
    IntegerAttr digitsAttr;
    if (nameRef == "log" && argObjs.size() == 2) {
      auto baseVal = getNumericLiteral(*argObjs[1]);
      if (!baseVal) {
        (void)emitError("log() base must be a numeric literal");
        return Value();
      }
      baseAttr = builder.getF64FloatAttr(*baseVal);
    }
    if ((nameRef == "round" || nameRef == "signif") && argObjs.size() == 2) {
      auto digitsVal = getNumericLiteral(*argObjs[1]);
      if (!digitsVal) {
        (void)emitError("digits must be a numeric literal");
        return Value();
      }
      double rounded = std::round(*digitsVal);
      if (std::abs(rounded - *digitsVal) > 1e-8) {
        (void)emitError("digits must be an integer literal");
        return Value();
      }
      digitsAttr = builder.getI64IntegerAttr(static_cast<int64_t>(rounded));
    }
    if ((nameRef == "round" || nameRef == "signif") && argObjs.size() == 1) {
      int64_t def = nameRef == "signif" ? 6 : 0;
      digitsAttr = builder.getI64IntegerAttr(def);
    }

    return r::MathOp::create(builder, loc, input.getType(), input,
                             builder.getStringAttr(nameRef), baseAttr, digitsAttr);
  }

  if (nameRef == "atan2") {
    if (argObjs.size() != 2) {
      (void)emitError("atan2() expects two arguments");
      return Value();
    }
    Value lhs = parseExpr(*argObjs[0]);
    Value rhs = parseExpr(*argObjs[1]);
    if (!lhs || !rhs) return Value();
    if (lhs.getType() != rhs.getType()) {
      (void)emitError("atan2() operands must have matching types");
      return Value();
    }
    Type elemType = lhs.getType();
    if (auto memrefType = llvm::dyn_cast<MemRefType>(elemType)) {
      elemType = memrefType.getElementType();
    }
    if (!llvm::isa<FloatType>(elemType)) {
      (void)emitError("atan2() expects floating-point operands");
      return Value();
    }
    return r::Math2Op::create(builder, loc, lhs.getType(), lhs, rhs,
                              builder.getStringAttr(nameRef));
  }

  auto func = module.lookupSymbol<func::FuncOp>(*name);
  if (!func) return Value();

  SmallVector<Value, 8> args;
  for (auto *argObj : argObjs) {
    args.push_back(parseExpr(*argObj));
  }

  auto results = func.getResultTypes();
  auto call = func::CallOp::create(builder, loc, *name, results, args);
  if (results.empty()) return Value();
  return call.getResult(0);
}

} // namespace rdslmlir

#include "rdslmlir/Runtime/RuntimeAstLowerer.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace rdslmlir {

AstLowerer::AstLowerer(MLIRContext &ctx, ModuleOp module)
    : context(ctx), builder(&ctx), module(module) {}

LogicalResult AstLowerer::lowerModule(const llvm::json::Object &obj) {
  auto *funcs = obj.getArray("functions");
  if (!funcs) return emitError("module missing 'functions'");

  for (auto &fnVal : *funcs) {
    auto *fnObj = fnVal.getAsObject();
    if (!fnObj) return emitError("function entry is not an object");
    if (failed(declareFunction(*fnObj))) return failure();
  }

  for (auto &fnVal : *funcs) {
    auto *fnObj = fnVal.getAsObject();
    if (!fnObj) return emitError("function entry is not an object");
    if (failed(defineFunction(*fnObj))) return failure();
  }

  return success();
}

LogicalResult AstLowerer::emitError(llvm::StringRef message) {
  llvm::errs() << "rdslmlir-runtime: " << message << "\n";
  return failure();
}

void AstLowerer::pushScope() { scopes.emplace_back(); }
void AstLowerer::popScope() { scopes.pop_back(); }

void AstLowerer::bind(llvm::StringRef name, Value value) {
  scopes.back().insert({name, value});
}

Value AstLowerer::lookup(llvm::StringRef name) {
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end()) return found->second;
  }
  return Value();
}

LogicalResult AstLowerer::declareFunction(const llvm::json::Object &obj) {
  auto name = obj.getString("name");
  if (!name) return emitError("function missing 'name'");

  SmallVector<Type, 8> paramTypes;
  auto *params = obj.getArray("params");
  if (params) {
    for (auto &paramVal : *params) {
      auto *paramObj = paramVal.getAsObject();
      if (!paramObj) return emitError("param entry is not an object");
      auto *typeVal = paramObj->get("type");
      if (!typeVal) return emitError("param missing 'type'");
      paramTypes.push_back(parseType(*typeVal));
    }
  }

  SmallVector<Type, 1> results;
  if (auto *retVal = obj.get("returnType")) {
    if (!retVal->getAsNull()) {
      if (auto *retObj = retVal->getAsObject()) {
        if (!retObj->empty()) {
          results.push_back(parseType(*retVal));
        }
      } else {
        results.push_back(parseType(*retVal));
      }
    }
  }

  auto funcType = builder.getFunctionType(paramTypes, results);
  builder.setInsertionPointToEnd(module.getBody());
  builder.create<func::FuncOp>(builder.getUnknownLoc(), *name, funcType);
  return success();
}

LogicalResult AstLowerer::defineFunction(const llvm::json::Object &obj) {
  auto name = obj.getString("name");
  if (!name) return emitError("function missing 'name'");
  auto func = module.lookupSymbol<func::FuncOp>(*name);
  if (!func) return emitError("function declaration not found");

  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  pushScope();

  auto *params = obj.getArray("params");
  if (params) {
    size_t idx = 0;
    for (auto &paramVal : *params) {
      auto *paramObj = paramVal.getAsObject();
      if (!paramObj) return emitError("param entry is not an object");
      auto pname = paramObj->getString("name");
      if (!pname) return emitError("param missing 'name'");
      bind(*pname, entry->getArgument(idx));
      ++idx;
    }
  }

  auto *body = obj.getArray("body");
  if (!body) return emitError("function missing 'body'");
  for (auto &stmtVal : *body) {
    auto *stmtObj = stmtVal.getAsObject();
    if (!stmtObj) return emitError("statement is not an object");
    if (failed(parseStmt(*stmtObj))) return failure();
  }

  if (entry->empty() || !entry->back().hasTrait<OpTrait::IsTerminator>()) {
    if (func.getFunctionType().getNumResults() != 0) {
      return emitError("missing return in function body");
    }
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
  }

  popScope();
  return success();
}

Type AstLowerer::parseType(const llvm::json::Value &val) {
  if (auto str = val.getAsString()) {
    if (*str == "i32") return builder.getI32Type();
    if (*str == "f64") return builder.getF64Type();
    if (*str == "bool") return builder.getI1Type();
    if (*str == "index") return builder.getIndexType();
  }

  if (auto obj = val.getAsObject()) {
    auto kind = obj->getString("type");
    if (kind && *kind == "tensor") {
      auto *elemVal = obj->get("elem");
      if (!elemVal) return builder.getF64Type();
      Type elemType = parseType(*elemVal);
      SmallVector<int64_t, 4> shape;
      if (auto *shapeArr = obj->getArray("shape")) {
        for (auto &dimVal : *shapeArr) {
          if (dimVal.getAsNull()) {
            shape.push_back(ShapedType::kDynamic);
            continue;
          }
          if (auto dimInt = dimVal.getAsInteger()) {
            shape.push_back(*dimInt);
            continue;
          }
          shape.push_back(ShapedType::kDynamic);
        }
      }
      if (shape.empty()) shape.push_back(ShapedType::kDynamic);

      auto storage = obj->getString("storage");
      if (storage && *storage == "tensor") {
        return RankedTensorType::get(shape, elemType);
      }
      return MemRefType::get(shape, elemType);
    }
  }

  return builder.getF64Type();
}

Value AstLowerer::parseExpr(const llvm::json::Object &obj) {
  auto kind = obj.getString("type");
  if (!kind) return Value();

  if (*kind == "Number") {
    Type dtype = builder.getF64Type();
    if (auto *dtypeVal = obj.get("dtype")) dtype = parseType(*dtypeVal);
    auto val = obj.get("value");
    if (!val) return Value();
    if (llvm::isa<FloatType>(dtype)) {
      auto num = val->getAsNumber().value_or(0.0);
      return builder.create<r::ConstantOp>(builder.getUnknownLoc(), dtype,
                                           builder.getFloatAttr(dtype, num));
    }
    if (llvm::isa<IndexType>(dtype)) {
      auto num = val->getAsInteger().value_or(0);
      return builder.create<r::ConstantOp>(builder.getUnknownLoc(), dtype,
                                           builder.getIndexAttr(num));
    }
    auto num = val->getAsInteger().value_or(0);
    return builder.create<r::ConstantOp>(builder.getUnknownLoc(), dtype,
                                         builder.getIntegerAttr(dtype, num));
  }

  if (*kind == "Bool") {
    Type dtype = builder.getI1Type();
    if (auto *dtypeVal = obj.get("dtype")) dtype = parseType(*dtypeVal);
    auto val = obj.getBoolean("value");
    if (!val) return Value();
    return builder.create<r::ConstantOp>(builder.getUnknownLoc(), dtype,
                                         builder.getBoolAttr(*val));
  }

  if (*kind == "Var") {
    auto name = obj.getString("name");
    if (!name) return Value();
    return lookup(*name);
  }

  if (*kind == "Unary") {
    auto op = obj.getString("op");
    auto *valObj = obj.getObject("value");
    if (!op || !valObj) return Value();
    Value inner = parseExpr(*valObj);
    if (*op == "+") return inner;
    if (*op == "-") {
      auto type = inner.getType();
      if (llvm::isa<FloatType>(type)) {
        auto zero = builder.create<r::ConstantOp>(builder.getUnknownLoc(), type,
                                                  builder.getFloatAttr(type, 0.0));
        return builder.create<r::SubOp>(builder.getUnknownLoc(), zero, inner);
      }
      auto zero = builder.create<r::ConstantOp>(builder.getUnknownLoc(), type,
                                                builder.getIntegerAttr(type, 0));
      return builder.create<r::SubOp>(builder.getUnknownLoc(), zero, inner);
    }
    return Value();
  }

  if (*kind == "Binary") {
    auto op = obj.getString("op");
    auto *lhsObj = obj.getObject("lhs");
    auto *rhsObj = obj.getObject("rhs");
    if (!op || !lhsObj || !rhsObj) return Value();
    Value lhs = parseExpr(*lhsObj);
    Value rhs = parseExpr(*rhsObj);
    if (*op == "+") return builder.create<r::AddOp>(builder.getUnknownLoc(), lhs, rhs);
    if (*op == "-") return builder.create<r::SubOp>(builder.getUnknownLoc(), lhs, rhs);
    if (*op == "*") return builder.create<r::MulOp>(builder.getUnknownLoc(), lhs, rhs);
    if (*op == "/") return builder.create<r::DivOp>(builder.getUnknownLoc(), lhs, rhs);
    return Value();
  }

  if (*kind == "AllocTensor") {
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
    return builder.create<bufferization::AllocTensorOp>(builder.getUnknownLoc(), tensorType, sizes);
  }

  if (*kind == "Clone") {
    auto *sourceObj = obj.getObject("source");
    if (!sourceObj) return Value();
    Value source = parseExpr(*sourceObj);
    return builder.create<bufferization::CloneOp>(builder.getUnknownLoc(), source);
  }

  if (*kind == "ToTensor") {
    auto *sourceObj = obj.getObject("source");
    auto *typeVal = obj.get("tensorType");
    if (!sourceObj || !typeVal) return Value();
    Value source = parseExpr(*sourceObj);
    Type tensorType = parseType(*typeVal);
    if (!llvm::isa<RankedTensorType>(tensorType)) return Value();
    return builder.create<bufferization::ToTensorOp>(builder.getUnknownLoc(), tensorType, source);
  }

  if (*kind == "ToBuffer") {
    auto *sourceObj = obj.getObject("source");
    auto *typeVal = obj.get("bufferType");
    if (!sourceObj || !typeVal) return Value();
    Value source = parseExpr(*sourceObj);
    Type bufferType = parseType(*typeVal);
    if (!llvm::isa<MemRefType>(bufferType)) return Value();
    return builder.create<bufferization::ToBufferOp>(builder.getUnknownLoc(), bufferType, source);
  }

  if (*kind == "MaterializeInDestination") {
    auto *sourceObj = obj.getObject("source");
    auto *destObj = obj.getObject("dest");
    if (!sourceObj || !destObj) return Value();
    Value source = parseExpr(*sourceObj);
    Value dest = parseExpr(*destObj);
    auto loc = builder.getUnknownLoc();
    if (llvm::isa<BaseMemRefType>(dest.getType())) {
      builder.create<bufferization::MaterializeInDestinationOp>(
          loc, Type(), source, dest, /*restrict=*/false, /*writable=*/true);
      return dest;
    }
    auto op = builder.create<bufferization::MaterializeInDestinationOp>(loc, source, dest);
    if (op.getNumResults() == 0) return dest;
    return op.getResult();
  }

  if (*kind == "Dim") {
    auto *targetObj = obj.getObject("target");
    auto axis = obj.getInteger("axis");
    if (!targetObj || !axis) return Value();
    Value target = parseExpr(*targetObj);
    if (!llvm::isa<MemRefType>(target.getType())) return Value();
    return builder.create<memref::DimOp>(builder.getUnknownLoc(), target, *axis);
  }

  if (*kind == "Index") {
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
    return builder.create<r::LoadOp>(builder.getUnknownLoc(), memrefType.getElementType(),
                                     target, indices);
  }

  if (*kind == "Slice") {
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
      auto one = builder.create<r::ConstantOp>(builder.getUnknownLoc(), builder.getIndexType(),
                                               builder.getIndexAttr(1));
      strides.assign(offsets.size(), one);
    }

    int64_t rank = memrefType.getRank();
    if ((int64_t)offsets.size() != rank || (int64_t)sizes.size() != rank || (int64_t)strides.size() != rank) {
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
    return builder.create<r::SliceOp>(builder.getUnknownLoc(), resultType, source, offsets, sizes,
                                      strides, dropAttr);
  }

  if (*kind == "MatMul") {
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
    return builder.create<r::MatmulOp>(builder.getUnknownLoc(), resultType, lhs, rhs);
  }

  if (*kind == "Call") {
    auto name = obj.getString("name");
    if (!name) return Value();
    auto func = module.lookupSymbol<func::FuncOp>(*name);
    if (!func) return Value();

    SmallVector<Value, 8> args;
    if (auto *argsArr = obj.getArray("args")) {
      for (auto &argVal : *argsArr) {
        auto *argObj = argVal.getAsObject();
        if (!argObj) return Value();
        args.push_back(parseExpr(*argObj));
      }
    }

    auto results = func.getResultTypes();
    auto call = builder.create<func::CallOp>(builder.getUnknownLoc(), *name, results, args);
    if (results.empty()) return Value();
    return call.getResult(0);
  }

  return Value();
}

Value AstLowerer::ensureIndex(Value value) {
  if (llvm::isa<IndexType>(value.getType())) return value;
  if (llvm::isa<IntegerType>(value.getType())) {
    return builder.create<arith::IndexCastOp>(builder.getUnknownLoc(), builder.getIndexType(), value);
  }
  return value;
}

Value AstLowerer::coerceStoreValue(Value value, Type targetType) {
  if (!value) return value;
  Type srcType = value.getType();
  if (srcType == targetType) return value;

  Location loc = builder.getUnknownLoc();
  if (llvm::isa<IndexType>(srcType) && llvm::isa<IntegerType>(targetType)) {
    return builder.create<arith::IndexCastOp>(loc, targetType, value);
  }

  if (auto srcInt = llvm::dyn_cast<IntegerType>(srcType)) {
    if (llvm::isa<IndexType>(targetType)) {
      return builder.create<arith::IndexCastOp>(loc, targetType, value);
    }
    if (auto dstInt = llvm::dyn_cast<IntegerType>(targetType)) {
      unsigned srcWidth = srcInt.getWidth();
      unsigned dstWidth = dstInt.getWidth();
      if (srcWidth == dstWidth) return value;
      if (srcWidth < dstWidth) {
        return builder.create<arith::ExtSIOp>(loc, dstInt, value);
      }
      return builder.create<arith::TruncIOp>(loc, dstInt, value);
    }
  }

  return value;
}

LogicalResult AstLowerer::parseIndexTarget(const llvm::json::Object &obj, Value &memref,
                                           SmallVectorImpl<Value> &indices) {
  auto kind = obj.getString("type");
  if (!kind || *kind != "Index") return emitError("store target must be Index");
  auto *targetObj = obj.getObject("target");
  if (!targetObj) return emitError("index missing target");
  memref = parseExpr(*targetObj);
  if (!llvm::isa<MemRefType>(memref.getType())) return emitError("index target is not memref");

  if (auto *idxArr = obj.getArray("indices")) {
    for (auto &idxVal : *idxArr) {
      auto *idxObj = idxVal.getAsObject();
      if (!idxObj) return emitError("index entry is not object");
      indices.push_back(ensureIndex(parseExpr(*idxObj)));
    }
  }
  return success();
}

LogicalResult AstLowerer::parseStmt(const llvm::json::Object &obj) {
  auto kind = obj.getString("type");
  if (!kind) return emitError("statement missing type");

  if (*kind == "Assign") {
    auto name = obj.getString("name");
    auto *valObj = obj.getObject("value");
    if (!name || !valObj) return emitError("assign missing fields");
    Value value = parseExpr(*valObj);
    bind(*name, value);
    return success();
  }

  if (*kind == "Dealloc") {
    auto *targetObj = obj.getObject("target");
    if (!targetObj) return emitError("dealloc missing target");
    Value target = parseExpr(*targetObj);
    builder.create<bufferization::DeallocOp>(builder.getUnknownLoc(), target);
    return success();
  }

  if (*kind == "DeallocTensor") {
    auto *targetObj = obj.getObject("target");
    if (!targetObj) return emitError("dealloc_tensor missing target");
    Value target = parseExpr(*targetObj);
    builder.create<bufferization::DeallocTensorOp>(builder.getUnknownLoc(), target);
    return success();
  }

  if (*kind == "Store") {
    auto *targetObj = obj.getObject("target");
    auto *valueObj = obj.getObject("value");
    if (!targetObj || !valueObj) return emitError("store missing fields");
    Value memref;
    SmallVector<Value, 4> indices;
    if (failed(parseIndexTarget(*targetObj, memref, indices))) return failure();
    Value value = parseExpr(*valueObj);
    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) return emitError("store target is not memref");
    value = coerceStoreValue(value, memrefType.getElementType());
    builder.create<r::StoreOp>(builder.getUnknownLoc(), value, memref, indices);
    return success();
  }

  if (*kind == "For") {
    auto idxName = obj.getString("index");
    auto *startObj = obj.getObject("start");
    auto *endObj = obj.getObject("end");
    auto *stepObj = obj.getObject("step");
    auto *body = obj.getArray("body");
    if (!idxName || !startObj || !endObj || !stepObj || !body) {
      return emitError("for missing fields");
    }

    Value startVal = ensureIndex(parseExpr(*startObj));
    Value endVal = ensureIndex(parseExpr(*endObj));
    Value stepVal = ensureIndex(parseExpr(*stepObj));

    bool parallel = false;
    if (auto p = obj.getBoolean("parallel")) parallel = *p;

    Location loc = builder.getUnknownLoc();
    Operation *loopOp = nullptr;
    if (parallel) {
      loopOp = builder.create<r::ParallelForOp>(loc, startVal, endVal, stepVal);
    } else {
      loopOp = builder.create<r::ForOp>(loc, startVal, endVal, stepVal);
    }

    Region &region = loopOp->getRegion(0);
    Block *block = new Block();
    region.push_back(block);
    block->addArgument(builder.getIndexType(), loc);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    pushScope();
    bind(*idxName, block->getArgument(0));

    for (auto &stmtVal : *body) {
      auto *stmtObj = stmtVal.getAsObject();
      if (!stmtObj) return emitError("for body statement is not object");
      if (failed(parseStmt(*stmtObj))) return failure();
    }
    builder.create<r::YieldOp>(loc);
    popScope();
    return success();
  }

  if (*kind == "Return") {
    if (auto *valueObj = obj.getObject("value")) {
      Value value = parseExpr(*valueObj);
      builder.create<func::ReturnOp>(builder.getUnknownLoc(), value);
    } else {
      builder.create<func::ReturnOp>(builder.getUnknownLoc());
    }
    return success();
  }

  return emitError("unknown statement type");
}

} // namespace rdslmlir

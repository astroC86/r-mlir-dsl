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
  func::FuncOp::create(builder, builder.getUnknownLoc(), *name, funcType);
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
    func::ReturnOp::create(builder, builder.getUnknownLoc());
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


Value AstLowerer::ensureIndex(Value value) {
  if (llvm::isa<IndexType>(value.getType())) return value;
  if (llvm::isa<IntegerType>(value.getType())) {
    return arith::IndexCastOp::create(builder, builder.getUnknownLoc(), builder.getIndexType(), value);
  }
  return value;
}

Value AstLowerer::coerceStoreValue(Value value, Type targetType) {
  if (!value) return value;
  Type srcType = value.getType();
  if (srcType == targetType) return value;

  Location loc = builder.getUnknownLoc();
  if (llvm::isa<IndexType>(srcType) && llvm::isa<IntegerType>(targetType)) {
    return arith::IndexCastOp::create(builder, loc, targetType, value);
  }

  if (auto srcInt = llvm::dyn_cast<IntegerType>(srcType)) {
    if (llvm::isa<IndexType>(targetType)) {
      return arith::IndexCastOp::create(builder, loc, targetType, value);
    }
    if (auto dstInt = llvm::dyn_cast<IntegerType>(targetType)) {
      unsigned srcWidth = srcInt.getWidth();
      unsigned dstWidth = dstInt.getWidth();
      if (srcWidth == dstWidth) return value;
      if (srcWidth < dstWidth) {
        return arith::ExtSIOp::create(builder, loc, dstInt, value);
      }
      return arith::TruncIOp::create(builder, loc, dstInt, value);
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


} // namespace rdslmlir

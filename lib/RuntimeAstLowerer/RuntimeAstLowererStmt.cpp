#include "rdslmlir/Runtime/RuntimeAstLowerer.h"

using namespace mlir;

namespace rdslmlir {

const llvm::StringMap<AstLowerer::StmtHandler> &AstLowerer::getStmtHandlers() {
  static const llvm::StringMap<StmtHandler> handlers = [] {
    llvm::StringMap<StmtHandler> map;
    map.try_emplace("Assign", &AstLowerer::parseAssignStmt);
    map.try_emplace("Dealloc", &AstLowerer::parseDeallocStmt);
    map.try_emplace("DeallocTensor", &AstLowerer::parseDeallocTensorStmt);
    map.try_emplace("Store", &AstLowerer::parseStoreStmt);
    map.try_emplace("For", &AstLowerer::parseForStmt);
    map.try_emplace("Return", &AstLowerer::parseReturnStmt);
    return map;
  }();
  return handlers;
}

LogicalResult AstLowerer::parseStmt(const llvm::json::Object &obj) {
  auto kind = obj.getString("type");
  if (!kind) return emitError("statement missing type");
  auto &handlers = getStmtHandlers();
  auto it = handlers.find(*kind);
  if (it == handlers.end()) return emitError("unknown statement type");
  return (this->*(it->second))(obj);
}

LogicalResult AstLowerer::parseAssignStmt(const llvm::json::Object &obj) {
  auto name = obj.getString("name");
  auto *valObj = obj.getObject("value");
  if (!name || !valObj) return emitError("assign missing fields");
  Value value = parseExpr(*valObj);
  bind(*name, value);
  return success();
}

LogicalResult AstLowerer::parseDeallocStmt(const llvm::json::Object &obj) {
  auto *targetObj = obj.getObject("target");
  if (!targetObj) return emitError("dealloc missing target");
  Value target = parseExpr(*targetObj);
  bufferization::DeallocOp::create(builder, builder.getUnknownLoc(), target);
  return success();
}

LogicalResult AstLowerer::parseDeallocTensorStmt(const llvm::json::Object &obj) {
  auto *targetObj = obj.getObject("target");
  if (!targetObj) return emitError("dealloc_tensor missing target");
  Value target = parseExpr(*targetObj);
  bufferization::DeallocTensorOp::create(builder, builder.getUnknownLoc(), target);
  return success();
}

LogicalResult AstLowerer::parseStoreStmt(const llvm::json::Object &obj) {
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
  r::StoreOp::create(builder, builder.getUnknownLoc(), value, memref, indices);
  return success();
}

LogicalResult AstLowerer::parseForStmt(const llvm::json::Object &obj) {
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
    loopOp = r::ParallelForOp::create(builder, loc, startVal, endVal, stepVal);
  } else {
    loopOp = r::ForOp::create(builder, loc, startVal, endVal, stepVal);
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
  r::YieldOp::create(builder, loc);
  popScope();
  return success();
}

LogicalResult AstLowerer::parseReturnStmt(const llvm::json::Object &obj) {
  if (auto *valueObj = obj.getObject("value")) {
    Value value = parseExpr(*valueObj);
    func::ReturnOp::create(builder, builder.getUnknownLoc(), value);
  } else {
    func::ReturnOp::create(builder, builder.getUnknownLoc());
  }
  return success();
}

} // namespace rdslmlir

#ifndef RDSLMLIR_RUNTIME_RUNTIME_AST_LOWERER_H
#define RDSLMLIR_RUNTIME_RUNTIME_AST_LOWERER_H

#include "rdslmlir/Dialect/R/IR/ROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"

namespace rdslmlir {

class AstLowerer {
public:
  AstLowerer(mlir::MLIRContext &ctx, mlir::ModuleOp module);

  mlir::LogicalResult lowerModule(const llvm::json::Object &obj);

private:
  mlir::MLIRContext &context;
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
  llvm::SmallVector<llvm::StringMap<mlir::Value>, 8> scopes;

  // Diagnostics and scope management
  mlir::LogicalResult emitError(llvm::StringRef message);
  void pushScope();
  void popScope();
  void bind(llvm::StringRef name, mlir::Value value);
  mlir::Value lookup(llvm::StringRef name);

  // Module/function lowering
  mlir::LogicalResult declareFunction(const llvm::json::Object &obj);
  mlir::LogicalResult defineFunction(const llvm::json::Object &obj);

  // Type parsing
  mlir::Type parseType(const llvm::json::Value &val);

  // Expression parsing
  using ExprHandler = mlir::Value (AstLowerer::*)(const llvm::json::Object &);
  mlir::Value parseExpr(const llvm::json::Object &obj);
  static const llvm::StringMap<ExprHandler> &getExprHandlers();
  mlir::Value parseNumberExpr(const llvm::json::Object &obj);
  mlir::Value parseBoolExpr(const llvm::json::Object &obj);
  mlir::Value parseVarExpr(const llvm::json::Object &obj);
  mlir::Value parseUnaryExpr(const llvm::json::Object &obj);
  mlir::Value parseBinaryExpr(const llvm::json::Object &obj);
  mlir::Value parseAllocTensorExpr(const llvm::json::Object &obj);
  mlir::Value parseCloneExpr(const llvm::json::Object &obj);
  mlir::Value parseToTensorExpr(const llvm::json::Object &obj);
  mlir::Value parseToBufferExpr(const llvm::json::Object &obj);
  mlir::Value parseMaterializeInDestinationExpr(const llvm::json::Object &obj);
  mlir::Value parseDimExpr(const llvm::json::Object &obj);
  mlir::Value parseIndexExpr(const llvm::json::Object &obj);
  mlir::Value parseSliceExpr(const llvm::json::Object &obj);
  mlir::Value parseMatMulExpr(const llvm::json::Object &obj);
  mlir::Value parseCallExpr(const llvm::json::Object &obj);

  // Statement parsing
  using StmtHandler = mlir::LogicalResult (AstLowerer::*)(const llvm::json::Object &);
  mlir::LogicalResult parseStmt(const llvm::json::Object &obj);
  static const llvm::StringMap<StmtHandler> &getStmtHandlers();
  mlir::LogicalResult parseAssignStmt(const llvm::json::Object &obj);
  mlir::LogicalResult parseDeallocStmt(const llvm::json::Object &obj);
  mlir::LogicalResult parseDeallocTensorStmt(const llvm::json::Object &obj);
  mlir::LogicalResult parseStoreStmt(const llvm::json::Object &obj);
  mlir::LogicalResult parseForStmt(const llvm::json::Object &obj);
  mlir::LogicalResult parseReturnStmt(const llvm::json::Object &obj);

  // Helpers
  mlir::Value ensureIndex(mlir::Value value);
  mlir::Value coerceStoreValue(mlir::Value value, mlir::Type targetType);
  mlir::LogicalResult parseIndexTarget(const llvm::json::Object &obj, mlir::Value &memref,
                                       llvm::SmallVectorImpl<mlir::Value> &indices);
};

} // namespace rdslmlir

#endif // RDSLMLIR_RUNTIME_RUNTIME_AST_LOWERER_H

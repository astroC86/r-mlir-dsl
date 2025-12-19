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

  mlir::LogicalResult emitError(llvm::StringRef message);
  void pushScope();
  void popScope();
  void bind(llvm::StringRef name, mlir::Value value);
  mlir::Value lookup(llvm::StringRef name);

  mlir::LogicalResult declareFunction(const llvm::json::Object &obj);
  mlir::LogicalResult defineFunction(const llvm::json::Object &obj);
  mlir::Type parseType(const llvm::json::Value &val);
  mlir::Value parseExpr(const llvm::json::Object &obj);
  mlir::Value ensureIndex(mlir::Value value);
  mlir::Value coerceStoreValue(mlir::Value value, mlir::Type targetType);
  mlir::LogicalResult parseIndexTarget(const llvm::json::Object &obj, mlir::Value &memref,
                                       llvm::SmallVectorImpl<mlir::Value> &indices);
  mlir::LogicalResult parseStmt(const llvm::json::Object &obj);
};

} // namespace rdslmlir

#endif // RDSLMLIR_RUNTIME_RUNTIME_AST_LOWERER_H

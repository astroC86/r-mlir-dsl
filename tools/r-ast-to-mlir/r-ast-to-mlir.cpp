#include "rdslmlir/Runtime/RuntimeAstLowerer.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input json>"),
                                           llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argc, argv, "R AST to MLIR\n");

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "r-ast-to-mlir: unable to read input\n";
    return 1;
  }

  auto json = llvm::json::parse(fileOrErr->get()->getBuffer());
  if (!json) {
    llvm::errs() << "r-ast-to-mlir: invalid JSON\n";
    return 1;
  }
  auto *root = json->getAsObject();
  if (!root) {
    llvm::errs() << "r-ast-to-mlir: root is not an object\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<r::RDialect, func::FuncDialect, arith::ArithDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  scf::SCFDialect, tensor::TensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());

  rdslmlir::AstLowerer lowerer(context, module);
  if (failed(lowerer.lowerModule(*root))) return 1;

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}

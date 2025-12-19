#include "rdslmlir/Conversion/RToLinalg/LoweringPatterns.h"
#include "rdslmlir/Conversion/RPasses.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct RLowerToSCFPass : PassWrapper<RLowerToSCFPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "r-lower-to-scf"; }
  StringRef getDescription() const final {
    return "Lower R dialect to SCF/arith/memref";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<r::RDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    r::populateRLowerToSCFPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct RLowerToLinalgPass : PassWrapper<RLowerToLinalgPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "r-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower R dialect to Linalg/memref";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<r::RDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    r::populateRLowerToLinalgPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> r::createRLowerToSCFPass() {
  return std::make_unique<RLowerToSCFPass>();
}

std::unique_ptr<Pass> r::createRLowerToLinalgPass() {
  return std::make_unique<RLowerToLinalgPass>();
}

void r::registerRPasses() {
  PassRegistration<RLowerToSCFPass>();
  PassRegistration<RLowerToLinalgPass>();

  PassPipelineRegistration<>(
      "r-to-scf",
      "Lower R dialect to SCF/arith/memref",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
      });

  PassPipelineRegistration<>(
      "r-to-linalg",
      "Lower R dialect to Linalg/memref",
      [](OpPassManager &pm) { pm.addPass(createRLowerToLinalgPass()); });

  PassPipelineRegistration<>(
      "r-to-openmp",
      "Lower R dialect and map parallel loops to OpenMP",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
        pm.addPass(mlir::createConvertSCFToOpenMPPass());
      });

  PassPipelineRegistration<>(
      "r-to-gpu",
      "Lower R dialect and map parallel loops to GPU",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
        pm.addPass(mlir::createConvertParallelLoopToGpuPass());
      });
}

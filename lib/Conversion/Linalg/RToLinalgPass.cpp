#include "rdslmlir/Conversion/Linalg/RToLinalgPass.h"

#include "rdslmlir/Conversion/Arith/RToArith.h"
#include "rdslmlir/Conversion/Linalg/RToLinalg.h"
#include "rdslmlir/Conversion/Math/RToMath.h"
#include "rdslmlir/Conversion/MemRef/RToMemRef.h"
#include "rdslmlir/Conversion/SCF/RToSCF.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct RLowerToLinalgPass : PassWrapper<RLowerToLinalgPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "r-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower R dialect to Linalg/memref";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<r::RDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                           math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    r::populateRLowerToArithLinalgPatterns(patterns);
    r::populateRLowerToMathLinalgPatterns(patterns);
    r::populateRLowerToMemRefPatterns(patterns);
    r::populateRLowerToSCFPatterns(patterns);
    r::populateRLowerToLinalgPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> r::createRLowerToLinalgPass() {
  return std::make_unique<RLowerToLinalgPass>();
}

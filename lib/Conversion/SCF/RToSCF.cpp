#include "rdslmlir/Conversion/SCF/RToSCF.h"

#include "rdslmlir/Conversion/Common/LoweringUtils.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct MatmulToSCFLowering : OpConversionPattern<r::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto lhsType = llvm::dyn_cast<MemRefType>(adaptor.getLhs().getType());
    auto rhsType = llvm::dyn_cast<MemRefType>(adaptor.getRhs().getType());
    auto resultType = llvm::dyn_cast<MemRefType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType) return failure();

    Value lowered = r::lowering::lowerMatmulToSCF(
        op.getLoc(), lhsType, rhsType, resultType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    if (!lowered) return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct ForLowering : OpConversionPattern<r::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value upper = adaptor.getUpper();
    Value one = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
    upper = arith::AddIOp::create(rewriter, op.getLoc(), upper, one);
    auto newFor = scf::ForOp::create(rewriter, op.getLoc(), adaptor.getLower(), upper,
                                     adaptor.getStep());
    Block *oldBody = &op.getBody().front();
    Block *newBody = newFor.getBody();

    rewriter.eraseOp(oldBody->getTerminator());
    rewriter.eraseOp(newBody->getTerminator());
    rewriter.mergeBlocks(oldBody, newBody, {newFor.getInductionVar()});
    rewriter.setInsertionPointToEnd(newBody);
    scf::YieldOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, newFor.getResults());
    return success();
  }
};

struct ParallelForLowering : OpConversionPattern<r::ParallelForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::ParallelForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value upper = adaptor.getUpper();
    Value one = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
    upper = arith::AddIOp::create(rewriter, op.getLoc(), upper, one);
    auto newPar = scf::ParallelOp::create(rewriter, op.getLoc(), ValueRange{adaptor.getLower()},
                                          ValueRange{upper},
                                          ValueRange{adaptor.getStep()});
    Block *oldBody = &op.getBody().front();
    Block *newBody = newPar.getBody();

    rewriter.eraseOp(oldBody->getTerminator());
    rewriter.inlineBlockBefore(oldBody, newBody->getTerminator(),
                               {newPar.getInductionVars().front()});
    rewriter.replaceOp(op, newPar.getResults());
    return success();
  }
};

struct YieldLowering : OpConversionPattern<r::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op);
    return success();
  }
};

} // namespace

void r::populateRLowerToSCFPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<MatmulToSCFLowering, ForLowering, ParallelForLowering, YieldLowering>(ctx);
}

#include "rdslmlir/Conversion/Arith/RToArith.h"

#include "rdslmlir/Conversion/Common/LoweringUtils.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct ConstantLowering : OpConversionPattern<r::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getResult().getType(), adaptor.getValue());
    return success();
  }
};

template <typename ROp, typename FloatOp, typename IntOp>
struct BinaryLowering : OpConversionPattern<ROp> {
  using OpConversionPattern<ROp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ROp op, typename ROp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      Value lowered = r::lowering::lowerElementwiseBinary<FloatOp, IntOp>(
          loc, memrefType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    Value lowered = r::lowering::createScalarBinary<FloatOp, IntOp>(
        loc, outType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    if (!lowered) return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

template <typename ROp, typename FloatOp, typename IntOp>
struct BinaryToLinalgLowering : OpConversionPattern<ROp> {
  using OpConversionPattern<ROp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ROp op, typename ROp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      Value lowered = r::lowering::lowerElementwiseBinaryToLinalg<FloatOp, IntOp>(
          loc, memrefType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    Value lowered = r::lowering::createScalarBinary<FloatOp, IntOp>(
        loc, outType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
    if (!lowered) return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

} // namespace

void r::populateRLowerToArithSCFPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ConstantLowering>(ctx);
  patterns.add<BinaryLowering<r::AddOp, arith::AddFOp, arith::AddIOp>>(ctx);
  patterns.add<BinaryLowering<r::SubOp, arith::SubFOp, arith::SubIOp>>(ctx);
  patterns.add<BinaryLowering<r::MulOp, arith::MulFOp, arith::MulIOp>>(ctx);
  patterns.add<BinaryLowering<r::DivOp, arith::DivFOp, arith::DivSIOp>>(ctx);
}

void r::populateRLowerToArithLinalgPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ConstantLowering>(ctx);
  patterns.add<BinaryToLinalgLowering<r::AddOp, arith::AddFOp, arith::AddIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::SubOp, arith::SubFOp, arith::SubIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::MulOp, arith::MulFOp, arith::MulIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::DivOp, arith::DivFOp, arith::DivSIOp>>(ctx);
}

#include "rdslmlir/Conversion/Linalg/RToLinalg.h"

#include "rdslmlir/Conversion/Common/LoweringUtils.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct MatmulLowering : OpConversionPattern<r::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto lhsType = llvm::dyn_cast<MemRefType>(adaptor.getLhs().getType());
    auto rhsType = llvm::dyn_cast<MemRefType>(adaptor.getRhs().getType());
    auto resultType = llvm::dyn_cast<MemRefType>(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType) return failure();

    SmallVector<Value, 2> dynSizes;
    if (resultType.isDynamicDim(0)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, op.getLoc(), adaptor.getLhs(), 0));
    }
    if (resultType.isDynamicDim(1)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, op.getLoc(), adaptor.getRhs(), 1));
    }

    Value output = memref::AllocOp::create(rewriter, op.getLoc(), resultType, dynSizes);
    Value zero = r::lowering::createZeroConstant(op.getLoc(), resultType.getElementType(), rewriter);
    if (!zero) return failure();
    linalg::FillOp::create(rewriter, op.getLoc(), zero, output);
    linalg::MatmulOp::create(rewriter, op.getLoc(), ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                             ValueRange{output});
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void r::populateRLowerToLinalgPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<MatmulLowering>(ctx);
}

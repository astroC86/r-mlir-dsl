#include "rdslmlir/Conversion/MemRef/RToMemRef.h"

#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct LoadLowering : OpConversionPattern<r::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getMemref(), adaptor.getIndices());
    return success();
  }
};

struct StoreLowering : OpConversionPattern<r::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getValue(), adaptor.getMemref(),
                                                adaptor.getIndices());
    return success();
  }
};

struct SliceLowering : OpConversionPattern<r::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto sourceType = llvm::cast<MemRefType>(adaptor.getSource().getType());
    int64_t rank = sourceType.getRank();
    SmallVector<int64_t, 4> staticOffsets(rank, ShapedType::kDynamic);
    SmallVector<int64_t, 4> staticSizes(rank, ShapedType::kDynamic);
    SmallVector<int64_t, 4> staticStrides(rank, ShapedType::kDynamic);
    SmallVector<bool, 4> dropDims(rank, false);
    if (auto dropAttr = op->getAttrOfType<DenseBoolArrayAttr>("drop_dims")) {
      auto dropVals = dropAttr.asArrayRef();
      if (static_cast<int64_t>(dropVals.size()) == rank) {
        for (int64_t i = 0; i < rank; ++i) {
          if (dropVals[i]) {
            staticSizes[i] = 1;
            dropDims[i] = true;
          }
        }
      }
    }

    auto resultType = llvm::cast<MemRefType>(op.getResult().getType());
    SmallVector<int64_t, 4> resultShape(resultType.getShape().begin(), resultType.getShape().end());
    SmallVector<OpFoldResult, 4> mixedOffsets;
    SmallVector<OpFoldResult, 4> mixedSizes;
    SmallVector<OpFoldResult, 4> mixedStrides;
    mixedOffsets.reserve(rank);
    mixedSizes.reserve(rank);
    mixedStrides.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      mixedOffsets.push_back(adaptor.getOffsets()[i]);
      if (dropDims[i]) {
        mixedSizes.push_back(rewriter.getIndexAttr(1));
        mixedStrides.push_back(rewriter.getIndexAttr(1));
      } else {
        mixedSizes.push_back(adaptor.getSizes()[i]);
        mixedStrides.push_back(adaptor.getStrides()[i]);
      }
    }

    MemRefType inferredType = memref::SubViewOp::inferRankReducedResultType(
        resultShape, sourceType, mixedOffsets, mixedSizes, mixedStrides);
    if (!inferredType) return failure();

    auto subview = memref::SubViewOp::create(rewriter, op.getLoc(), inferredType,
                                             adaptor.getSource(), mixedOffsets, mixedSizes,
                                             mixedStrides);
    rewriter.replaceOp(op, subview.getResult());
    return success();
  }
};

} // namespace

void r::populateRLowerToMemRefPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<LoadLowering, StoreLowering, SliceLowering>(ctx);
}

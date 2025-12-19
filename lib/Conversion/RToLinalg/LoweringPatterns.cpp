#include "rdslmlir/Conversion/RToLinalg/LoweringPatterns.h"
#include "rdslmlir/Conversion/RToLinalg/LoweringUtils.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

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
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    upper = rewriter.create<arith::AddIOp>(op.getLoc(), upper, one);
    auto newFor = rewriter.create<scf::ForOp>(op.getLoc(), adaptor.getLower(), upper,
                                              adaptor.getStep());
    Block *oldBody = &op.getBody().front();
    Block *newBody = newFor.getBody();

    rewriter.eraseOp(oldBody->getTerminator());
    rewriter.eraseOp(newBody->getTerminator());
    rewriter.mergeBlocks(oldBody, newBody, {newFor.getInductionVar()});
    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<scf::YieldOp>(op.getLoc());
    rewriter.replaceOp(op, newFor.getResults());
    return success();
  }
};

struct ParallelForLowering : OpConversionPattern<r::ParallelForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::ParallelForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value upper = adaptor.getUpper();
    Value one = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    upper = rewriter.create<arith::AddIOp>(op.getLoc(), upper, one);
    auto newPar = rewriter.create<scf::ParallelOp>(op.getLoc(), ValueRange{adaptor.getLower()},
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

    auto subview = rewriter.create<memref::SubViewOp>(op.getLoc(), inferredType, adaptor.getSource(),
                                                     mixedOffsets, mixedSizes, mixedStrides);
    rewriter.replaceOp(op, subview.getResult());
    return success();
  }
};

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
      dynSizes.push_back(rewriter.create<memref::DimOp>(op.getLoc(), adaptor.getLhs(), 0));
    }
    if (resultType.isDynamicDim(1)) {
      dynSizes.push_back(rewriter.create<memref::DimOp>(op.getLoc(), adaptor.getRhs(), 1));
    }

    Value output = rewriter.create<memref::AllocOp>(op.getLoc(), resultType, dynSizes);
    Value zero = r::lowering::createZeroConstant(op.getLoc(), resultType.getElementType(), rewriter);
    if (!zero) return failure();
    rewriter.create<linalg::FillOp>(op.getLoc(), zero, output);
    rewriter.create<linalg::MatmulOp>(op.getLoc(), ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                                      ValueRange{output});
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void r::populateRLowerToSCFPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ConstantLowering, LoadLowering, StoreLowering, SliceLowering, MatmulToSCFLowering,
               ForLowering, ParallelForLowering, YieldLowering>(ctx);
  patterns.add<BinaryLowering<r::AddOp, arith::AddFOp, arith::AddIOp>>(ctx);
  patterns.add<BinaryLowering<r::SubOp, arith::SubFOp, arith::SubIOp>>(ctx);
  patterns.add<BinaryLowering<r::MulOp, arith::MulFOp, arith::MulIOp>>(ctx);
  patterns.add<BinaryLowering<r::DivOp, arith::DivFOp, arith::DivSIOp>>(ctx);
}

void r::populateRLowerToLinalgPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ConstantLowering, LoadLowering, StoreLowering, SliceLowering, MatmulLowering,
               ForLowering, ParallelForLowering, YieldLowering>(ctx);
  patterns.add<BinaryToLinalgLowering<r::AddOp, arith::AddFOp, arith::AddIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::SubOp, arith::SubFOp, arith::SubIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::MulOp, arith::MulFOp, arith::MulIOp>>(ctx);
  patterns.add<BinaryToLinalgLowering<r::DivOp, arith::DivFOp, arith::DivSIOp>>(ctx);
}

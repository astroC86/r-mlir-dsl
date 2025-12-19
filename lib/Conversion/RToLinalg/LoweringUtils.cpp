#include "rdslmlir/Conversion/RToLinalg/LoweringUtils.h"

using namespace mlir;

namespace r {
namespace lowering {

Value getDimValue(Location loc, Value memref, MemRefType type, int64_t dim,
                  ConversionPatternRewriter &rewriter) {
  if (type.isDynamicDim(dim)) {
    return rewriter.create<memref::DimOp>(loc, memref, dim);
  }
  return rewriter.create<arith::ConstantIndexOp>(loc, type.getDimSize(dim));
}

Value createZeroConstant(Location loc, Type type, ConversionPatternRewriter &rewriter) {
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    return rewriter.create<arith::ConstantOp>(loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
  }
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    return rewriter.create<arith::ConstantOp>(loc, intType, rewriter.getIntegerAttr(intType, 0));
  }
  return nullptr;
}

Value lowerMatmulToSCF(Location loc, MemRefType lhsType, MemRefType rhsType,
                       MemRefType resultType, Value lhs, Value rhs,
                       ConversionPatternRewriter &rewriter) {
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) return nullptr;
  if (lhsType.getElementType() != rhsType.getElementType()) return nullptr;

  Type elemType = lhsType.getElementType();
  if (!llvm::isa<FloatType>(elemType) && !llvm::isa<IntegerType>(elemType) &&
      !llvm::isa<IndexType>(elemType)) {
    return nullptr;
  }

  SmallVector<Value, 2> dynSizes;
  if (resultType.isDynamicDim(0)) {
    dynSizes.push_back(rewriter.create<memref::DimOp>(loc, lhs, 0));
  }
  if (resultType.isDynamicDim(1)) {
    dynSizes.push_back(rewriter.create<memref::DimOp>(loc, rhs, 1));
  }

  Value result = rewriter.create<memref::AllocOp>(loc, resultType, dynSizes);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value dimM = getDimValue(loc, lhs, lhsType, 0, rewriter);
  Value dimN = getDimValue(loc, rhs, rhsType, 1, rewriter);
  Value dimK = getDimValue(loc, lhs, lhsType, 1, rewriter);

  auto outer = rewriter.create<scf::ForOp>(loc, c0, dimM, c1);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(outer.getBody());

  auto inner = rewriter.create<scf::ForOp>(loc, c0, dimN, c1);
  rewriter.setInsertionPointToStart(inner.getBody());

  Value zero = createZeroConstant(loc, elemType, rewriter);
  if (!zero) return nullptr;

  auto reduce = rewriter.create<scf::ForOp>(loc, c0, dimK, c1, ValueRange{zero});
  rewriter.setInsertionPointToStart(reduce.getBody());

  Value i = outer.getInductionVar();
  Value j = inner.getInductionVar();
  Value k = reduce.getInductionVar();
  Value acc = reduce.getRegionIterArg(0);

  Value lhsVal = rewriter.create<memref::LoadOp>(loc, lhs, ValueRange{i, k});
  Value rhsVal = rewriter.create<memref::LoadOp>(loc, rhs, ValueRange{k, j});
  Value mul = createScalarBinary<arith::MulFOp, arith::MulIOp>(loc, elemType, lhsVal, rhsVal, rewriter);
  if (!mul) return nullptr;
  Value sum = createScalarBinary<arith::AddFOp, arith::AddIOp>(loc, elemType, acc, mul, rewriter);
  if (!sum) return nullptr;
  rewriter.create<scf::YieldOp>(loc, sum);

  rewriter.setInsertionPointAfter(reduce);
  rewriter.create<memref::StoreOp>(loc, reduce.getResult(0), result, ValueRange{i, j});
  return result;
}

} // namespace lowering
} // namespace r

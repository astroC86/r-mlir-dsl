#ifndef RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_UTILS_H
#define RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

namespace r {
namespace lowering {

using namespace mlir;

Value getDimValue(Location loc, Value memref, MemRefType type, int64_t dim,
                  ConversionPatternRewriter &rewriter);

Value createZeroConstant(Location loc, Type type, ConversionPatternRewriter &rewriter);

Value lowerMatmulToSCF(Location loc, MemRefType lhsType, MemRefType rhsType,
                       MemRefType resultType, Value lhs, Value rhs,
                       ConversionPatternRewriter &rewriter);

template <typename FloatOp, typename IntOp>
Value createScalarBinary(Location loc, Type type, Value lhs, Value rhs,
                         ConversionPatternRewriter &rewriter) {
  if (llvm::isa<FloatType>(type)) {
    return rewriter.create<FloatOp>(loc, lhs, rhs);
  }
  if (llvm::isa<IntegerType>(type) || llvm::isa<IndexType>(type)) {
    return rewriter.create<IntOp>(loc, lhs, rhs);
  }
  return nullptr;
}

template <typename FloatOp, typename IntOp>
Value lowerElementwiseBinary(Location loc, MemRefType type, Value lhs, Value rhs,
                             ConversionPatternRewriter &rewriter) {
  int64_t rank = type.getRank();
  if (rank < 1 || rank > 2) return nullptr;

  SmallVector<Value, 4> dynSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (type.isDynamicDim(i)) dynSizes.push_back(rewriter.create<memref::DimOp>(loc, lhs, i));
  }
  Value result = rewriter.create<memref::AllocOp>(loc, type, dynSizes);

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  Value dim0 = getDimValue(loc, lhs, type, 0, rewriter);
  auto outer = rewriter.create<scf::ForOp>(loc, c0, dim0, c1);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(outer.getBody());

  if (rank == 1) {
    Value idx0 = outer.getInductionVar();
    auto lhsVal = rewriter.create<memref::LoadOp>(loc, lhs, ValueRange{idx0});
    auto rhsVal = rewriter.create<memref::LoadOp>(loc, rhs, ValueRange{idx0});
    Value out = createScalarBinary<FloatOp, IntOp>(loc, type.getElementType(), lhsVal, rhsVal, rewriter);
    if (!out) return nullptr;
    rewriter.create<memref::StoreOp>(loc, out, result, ValueRange{idx0});
    return result;
  }

  Value dim1 = getDimValue(loc, lhs, type, 1, rewriter);
  auto inner = rewriter.create<scf::ForOp>(loc, c0, dim1, c1);
  rewriter.setInsertionPointToStart(inner.getBody());

  Value idx0 = outer.getInductionVar();
  Value idx1 = inner.getInductionVar();
  auto lhsVal = rewriter.create<memref::LoadOp>(loc, lhs, ValueRange{idx0, idx1});
  auto rhsVal = rewriter.create<memref::LoadOp>(loc, rhs, ValueRange{idx0, idx1});
  Value out = createScalarBinary<FloatOp, IntOp>(loc, type.getElementType(), lhsVal, rhsVal, rewriter);
  if (!out) return nullptr;
  rewriter.create<memref::StoreOp>(loc, out, result, ValueRange{idx0, idx1});
  return result;
}

template <typename FloatOp, typename IntOp>
Value lowerElementwiseBinaryToLinalg(Location loc, MemRefType type, Value lhs, Value rhs,
                                     ConversionPatternRewriter &rewriter) {
  int64_t rank = type.getRank();
  if (rank < 1) return nullptr;

  Type elemType = type.getElementType();
  if (!llvm::isa<FloatType>(elemType) && !llvm::isa<IntegerType>(elemType) &&
      !llvm::isa<IndexType>(elemType)) {
    return nullptr;
  }

  SmallVector<Value, 4> dynSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (type.isDynamicDim(i)) dynSizes.push_back(rewriter.create<memref::DimOp>(loc, lhs, i));
  }
  Value result = rewriter.create<memref::AllocOp>(loc, type, dynSizes);

  SmallVector<AffineMap, 4> maps;
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  maps.push_back(idMap);
  maps.push_back(idMap);
  maps.push_back(idMap);

  SmallVector<utils::IteratorType, 4> iteratorTypes(rank, utils::IteratorType::parallel);

  auto regionBuilder = [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value lhsVal = args[0];
    Value rhsVal = args[1];
    Value outVal = nullptr;
    if (llvm::isa<FloatType>(elemType)) {
      outVal = nestedBuilder.create<FloatOp>(nestedLoc, lhsVal, rhsVal);
    } else if (llvm::isa<IntegerType>(elemType) || llvm::isa<IndexType>(elemType)) {
      outVal = nestedBuilder.create<IntOp>(nestedLoc, lhsVal, rhsVal);
    }
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, outVal);
  };

  rewriter.create<linalg::GenericOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{result},
                                     maps, iteratorTypes, regionBuilder);
  return result;
}

} // namespace lowering
} // namespace r

#endif // RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_UTILS_H

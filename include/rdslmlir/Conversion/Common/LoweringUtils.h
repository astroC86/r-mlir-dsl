#ifndef RDSLMLIR_CONVERSION_COMMON_LOWERING_UTILS_H
#define RDSLMLIR_CONVERSION_COMMON_LOWERING_UTILS_H

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

template <typename BuilderFn>
Value lowerElementwiseUnary(Location loc, MemRefType type, Value input,
                            ConversionPatternRewriter &rewriter,
                            BuilderFn builderFn) {
  int64_t rank = type.getRank();
  if (rank < 1 || rank > 2) return nullptr;

  SmallVector<Value, 4> dynSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (type.isDynamicDim(i)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, loc, input, i));
    }
  }
  Value result = memref::AllocOp::create(rewriter, loc, type, dynSizes);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value dim0 = getDimValue(loc, input, type, 0, rewriter);
  auto outer = scf::ForOp::create(rewriter, loc, c0, dim0, c1);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(outer.getBody());

  if (rank == 1) {
    Value idx0 = outer.getInductionVar();
    Value inVal = memref::LoadOp::create(rewriter, loc, input, ValueRange{idx0});
    Value out = builderFn(rewriter, loc, inVal);
    if (!out) return nullptr;
    memref::StoreOp::create(rewriter, loc, out, result, ValueRange{idx0});
    return result;
  }

  Value dim1 = getDimValue(loc, input, type, 1, rewriter);
  auto inner = scf::ForOp::create(rewriter, loc, c0, dim1, c1);
  rewriter.setInsertionPointToStart(inner.getBody());

  Value idx0 = outer.getInductionVar();
  Value idx1 = inner.getInductionVar();
  Value inVal = memref::LoadOp::create(rewriter, loc, input, ValueRange{idx0, idx1});
  Value out = builderFn(rewriter, loc, inVal);
  if (!out) return nullptr;
  memref::StoreOp::create(rewriter, loc, out, result, ValueRange{idx0, idx1});
  return result;
}

template <typename BuilderFn>
Value lowerElementwiseUnaryToLinalg(Location loc, MemRefType type, Value input,
                                    ConversionPatternRewriter &rewriter,
                                    BuilderFn builderFn) {
  int64_t rank = type.getRank();
  if (rank < 1) return nullptr;

  SmallVector<Value, 4> dynSizes;
  for (int64_t i = 0; i < rank; ++i) {
    if (type.isDynamicDim(i)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, loc, input, i));
    }
  }
  Value result = memref::AllocOp::create(rewriter, loc, type, dynSizes);

  SmallVector<AffineMap, 4> maps;
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  maps.push_back(idMap);
  maps.push_back(idMap);

  SmallVector<utils::IteratorType, 4> iteratorTypes(rank, utils::IteratorType::parallel);

  auto regionBuilder = [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value outVal = builderFn(nestedBuilder, nestedLoc, args[0]);
    linalg::YieldOp::create(nestedBuilder, nestedLoc, outVal);
  };

  linalg::GenericOp::create(rewriter, loc, TypeRange{}, ValueRange{input},
                            ValueRange{result}, maps, iteratorTypes, regionBuilder);
  return result;
}

template <typename FloatOp, typename IntOp>
Value createScalarBinary(Location loc, Type type, Value lhs, Value rhs,
                         ConversionPatternRewriter &rewriter) {
  if (llvm::isa<FloatType>(type)) {
    return FloatOp::create(rewriter, loc, lhs, rhs);
  }
  if (llvm::isa<IntegerType>(type) || llvm::isa<IndexType>(type)) {
    return IntOp::create(rewriter, loc, lhs, rhs);
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
    if (type.isDynamicDim(i)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, loc, lhs, i));
    }
  }
  Value result = memref::AllocOp::create(rewriter, loc, type, dynSizes);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

  Value dim0 = getDimValue(loc, lhs, type, 0, rewriter);
  auto outer = scf::ForOp::create(rewriter, loc, c0, dim0, c1);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(outer.getBody());

  if (rank == 1) {
    Value idx0 = outer.getInductionVar();
    auto lhsVal = memref::LoadOp::create(rewriter, loc, lhs, ValueRange{idx0});
    auto rhsVal = memref::LoadOp::create(rewriter, loc, rhs, ValueRange{idx0});
    Value out = createScalarBinary<FloatOp, IntOp>(loc, type.getElementType(), lhsVal, rhsVal, rewriter);
    if (!out) return nullptr;
    memref::StoreOp::create(rewriter, loc, out, result, ValueRange{idx0});
    return result;
  }

  Value dim1 = getDimValue(loc, lhs, type, 1, rewriter);
  auto inner = scf::ForOp::create(rewriter, loc, c0, dim1, c1);
  rewriter.setInsertionPointToStart(inner.getBody());

  Value idx0 = outer.getInductionVar();
  Value idx1 = inner.getInductionVar();
  auto lhsVal = memref::LoadOp::create(rewriter, loc, lhs, ValueRange{idx0, idx1});
  auto rhsVal = memref::LoadOp::create(rewriter, loc, rhs, ValueRange{idx0, idx1});
  Value out = createScalarBinary<FloatOp, IntOp>(loc, type.getElementType(), lhsVal, rhsVal, rewriter);
  if (!out) return nullptr;
  memref::StoreOp::create(rewriter, loc, out, result, ValueRange{idx0, idx1});
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
    if (type.isDynamicDim(i)) {
      dynSizes.push_back(memref::DimOp::create(rewriter, loc, lhs, i));
    }
  }
  Value result = memref::AllocOp::create(rewriter, loc, type, dynSizes);

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
      outVal = FloatOp::create(nestedBuilder, nestedLoc, lhsVal, rhsVal);
    } else if (llvm::isa<IntegerType>(elemType) || llvm::isa<IndexType>(elemType)) {
      outVal = IntOp::create(nestedBuilder, nestedLoc, lhsVal, rhsVal);
    }
    linalg::YieldOp::create(nestedBuilder, nestedLoc, outVal);
  };

  linalg::GenericOp::create(rewriter, loc, TypeRange{}, ValueRange{lhs, rhs},
                            ValueRange{result}, maps, iteratorTypes, regionBuilder);
  return result;
}

} // namespace lowering
} // namespace r

#endif // RDSLMLIR_CONVERSION_COMMON_LOWERING_UTILS_H

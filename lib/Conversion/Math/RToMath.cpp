#include "rdslmlir/Conversion/Math/RToMath.h"

#include "rdslmlir/Conversion/Common/LoweringUtils.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <optional>

using namespace mlir;
using llvm::StringRef;

namespace {

std::optional<double> getBaseAttr(Operation *op) {
  if (auto attr = op->getAttrOfType<FloatAttr>("base")) {
    return attr.getValueAsDouble();
  }
  return std::nullopt;
}

std::optional<int64_t> getDigitsAttr(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("digits")) {
    return attr.getInt();
  }
  return std::nullopt;
}

Value buildSign(OpBuilder &builder, Location loc, Value operand) {
  Type type = operand.getType();
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    Value zero = arith::ConstantOp::create(builder, loc, floatType,
                                           builder.getFloatAttr(floatType, 0.0));
    Value one = arith::ConstantOp::create(builder, loc, floatType,
                                          builder.getFloatAttr(floatType, 1.0));
    Value negOne = arith::ConstantOp::create(builder, loc, floatType,
                                             builder.getFloatAttr(floatType, -1.0));
    Value isPos = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT, operand, zero);
    Value isNeg = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT, operand, zero);
    Value posVal = arith::SelectOp::create(builder, loc, isPos, one, zero);
    return arith::SelectOp::create(builder, loc, isNeg, negOne, posVal);
  }
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    Value zero = arith::ConstantOp::create(builder, loc, intType,
                                           builder.getIntegerAttr(intType, 0));
    Value one = arith::ConstantOp::create(builder, loc, intType,
                                          builder.getIntegerAttr(intType, 1));
    Value negOne = arith::ConstantOp::create(builder, loc, intType,
                                             builder.getIntegerAttr(intType, -1));
    Value isPos = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt, operand, zero);
    Value isNeg = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, operand, zero);
    Value posVal = arith::SelectOp::create(builder, loc, isPos, one, zero);
    return arith::SelectOp::create(builder, loc, isNeg, negOne, posVal);
  }
  return nullptr;
}

Value buildRound(OpBuilder &builder, Location loc, Value operand, int64_t digits) {
  auto floatType = llvm::dyn_cast<FloatType>(operand.getType());
  if (!floatType) return nullptr;

  if (digits == 0) {
    return math::RoundOp::create(builder, loc, operand);
  }

  double scaleVal = std::pow(10.0, static_cast<double>(digits));
  Value scale = arith::ConstantOp::create(builder, loc, floatType,
                                          builder.getFloatAttr(floatType, scaleVal));
  Value scaled = arith::MulFOp::create(builder, loc, operand, scale);
  Value rounded = math::RoundOp::create(builder, loc, scaled);
  return arith::DivFOp::create(builder, loc, rounded, scale);
}

Value buildSignif(OpBuilder &builder, Location loc, Value operand, int64_t digits) {
  auto floatType = llvm::dyn_cast<FloatType>(operand.getType());
  if (!floatType) return nullptr;

  Value zero = arith::ConstantOp::create(builder, loc, floatType,
                                         builder.getFloatAttr(floatType, 0.0));
  Value one = arith::ConstantOp::create(builder, loc, floatType,
                                        builder.getFloatAttr(floatType, 1.0));
  Value ten = arith::ConstantOp::create(builder, loc, floatType,
                                        builder.getFloatAttr(floatType, 10.0));
  Value digitsVal = arith::ConstantOp::create(builder, loc, floatType,
                                              builder.getFloatAttr(floatType, static_cast<double>(digits)));

  Value absVal = math::AbsFOp::create(builder, loc, operand);
  Value isZero = arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OEQ, absVal, zero);
  Value log10Val = math::Log10Op::create(builder, loc, absVal);
  Value floorVal = math::FloorOp::create(builder, loc, log10Val);
  Value expVal = arith::SubFOp::create(builder, loc, digitsVal, floorVal);
  Value expMinusOne = arith::SubFOp::create(builder, loc, expVal, one);
  Value scale = math::PowFOp::create(builder, loc, ten, expMinusOne);
  Value scaled = arith::MulFOp::create(builder, loc, operand, scale);
  Value rounded = math::RoundOp::create(builder, loc, scaled);
  Value result = arith::DivFOp::create(builder, loc, rounded, scale);
  return arith::SelectOp::create(builder, loc, isZero, zero, result);
}

Value buildMathUnary(OpBuilder &builder, Location loc, Value operand, StringRef kind,
                     std::optional<double> base, std::optional<int64_t> digits) {
  Type type = operand.getType();
  bool isFloat = llvm::isa<FloatType>(type);
  bool isInt = llvm::isa<IntegerType>(type);

  if (kind == "abs") {
    if (isFloat) return math::AbsFOp::create(builder, loc, operand);
    if (isInt) return math::AbsIOp::create(builder, loc, operand);
    return nullptr;
  }

  if (kind == "sign") {
    return buildSign(builder, loc, operand);
  }

  if (!isFloat) return nullptr;

  if (kind == "sqrt") return math::SqrtOp::create(builder, loc, operand);
  if (kind == "exp") return math::ExpOp::create(builder, loc, operand);
  if (kind == "log") {
    Value logVal = math::LogOp::create(builder, loc, operand);
    if (base) {
      auto floatType = llvm::cast<FloatType>(type);
      Value baseConst = arith::ConstantOp::create(builder, loc, floatType,
                                                  builder.getFloatAttr(floatType, *base));
      Value logBase = math::LogOp::create(builder, loc, baseConst);
      return arith::DivFOp::create(builder, loc, logVal, logBase);
    }
    return logVal;
  }
  if (kind == "log10") return math::Log10Op::create(builder, loc, operand);
  if (kind == "log2") return math::Log2Op::create(builder, loc, operand);
  if (kind == "sin") return math::SinOp::create(builder, loc, operand);
  if (kind == "cos") return math::CosOp::create(builder, loc, operand);
  if (kind == "tan") return math::TanOp::create(builder, loc, operand);
  if (kind == "asin") return math::AsinOp::create(builder, loc, operand);
  if (kind == "acos") return math::AcosOp::create(builder, loc, operand);
  if (kind == "atan") return math::AtanOp::create(builder, loc, operand);
  if (kind == "sinh") return math::SinhOp::create(builder, loc, operand);
  if (kind == "cosh") return math::CoshOp::create(builder, loc, operand);
  if (kind == "tanh") return math::TanhOp::create(builder, loc, operand);
  if (kind == "asinh") return math::AsinhOp::create(builder, loc, operand);
  if (kind == "acosh") return math::AcoshOp::create(builder, loc, operand);
  if (kind == "atanh") return math::AtanhOp::create(builder, loc, operand);
  if (kind == "floor") return math::FloorOp::create(builder, loc, operand);
  if (kind == "ceiling") return math::CeilOp::create(builder, loc, operand);
  if (kind == "trunc") return math::TruncOp::create(builder, loc, operand);
  if (kind == "round") return buildRound(builder, loc, operand, digits.value_or(0));
  if (kind == "signif") return buildSignif(builder, loc, operand, digits.value_or(6));

  return nullptr;
}

struct MathLowering : OpConversionPattern<r::MathOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::MathOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kindAttr = op->getAttrOfType<StringAttr>("kind");
    if (!kindAttr) return failure();
    StringRef kind = kindAttr.getValue();
    std::optional<double> base = getBaseAttr(op);
    std::optional<int64_t> digits = getDigitsAttr(op);

    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      Value lowered = r::lowering::lowerElementwiseUnary(
          loc, memrefType, adaptor.getInput(), rewriter,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value val) {
            return buildMathUnary(nestedBuilder, nestedLoc, val, kind, base, digits);
          });
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    Value lowered = buildMathUnary(rewriter, loc, adaptor.getInput(), kind, base, digits);
    if (!lowered) return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct MathToLinalgLowering : OpConversionPattern<r::MathOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::MathOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kindAttr = op->getAttrOfType<StringAttr>("kind");
    if (!kindAttr) return failure();
    StringRef kind = kindAttr.getValue();
    std::optional<double> base = getBaseAttr(op);
    std::optional<int64_t> digits = getDigitsAttr(op);

    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      Value lowered = r::lowering::lowerElementwiseUnaryToLinalg(
          loc, memrefType, adaptor.getInput(), rewriter,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, Value val) {
            return buildMathUnary(nestedBuilder, nestedLoc, val, kind, base, digits);
          });
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    Value lowered = buildMathUnary(rewriter, loc, adaptor.getInput(), kind, base, digits);
    if (!lowered) return failure();
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct Math2Lowering : OpConversionPattern<r::Math2Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::Math2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kindAttr = op->getAttrOfType<StringAttr>("kind");
    if (!kindAttr) return failure();
    if (kindAttr.getValue() != "atan2") return failure();

    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      if (!llvm::isa<FloatType>(memrefType.getElementType())) return failure();
      Value lowered = r::lowering::lowerElementwiseBinary<math::Atan2Op, math::Atan2Op>(
          loc, memrefType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    if (!llvm::isa<FloatType>(outType)) return failure();
    Value lowered = math::Atan2Op::create(rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

struct Math2ToLinalgLowering : OpConversionPattern<r::Math2Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(r::Math2Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kindAttr = op->getAttrOfType<StringAttr>("kind");
    if (!kindAttr) return failure();
    if (kindAttr.getValue() != "atan2") return failure();

    Type outType = op.getResult().getType();
    Location loc = op.getLoc();

    if (auto memrefType = llvm::dyn_cast<MemRefType>(outType)) {
      if (!llvm::isa<FloatType>(memrefType.getElementType())) return failure();
      Value lowered = r::lowering::lowerElementwiseBinaryToLinalg<math::Atan2Op, math::Atan2Op>(
          loc, memrefType, adaptor.getLhs(), adaptor.getRhs(), rewriter);
      if (!lowered) return failure();
      rewriter.replaceOp(op, lowered);
      return success();
    }

    if (!llvm::isa<FloatType>(outType)) return failure();
    Value lowered = math::Atan2Op::create(rewriter, loc, adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, lowered);
    return success();
  }
};

} // namespace

void r::populateRLowerToMathSCFPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<MathLowering, Math2Lowering>(ctx);
}

void r::populateRLowerToMathLinalgPatterns(RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<MathToLinalgLowering, Math2ToLinalgLowering>(ctx);
}

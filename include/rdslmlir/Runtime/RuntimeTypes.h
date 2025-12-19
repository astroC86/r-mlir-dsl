#ifndef RDSLMLIR_RUNTIME_RUNTIME_TYPES_H
#define RDSLMLIR_RUNTIME_RUNTIME_TYPES_H

#include <Rcpp.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rdslmlir {

enum class ScalarKind {
  kI32,
  kF64,
  kBool,
  kIndex,
  kUnsupported
};

struct MemRefInfo {
  ScalarKind elem;
  int rank;
  std::vector<int64_t> shape;
};

struct TypeInfo {
  bool isMemRef = false;
  ScalarKind scalar = ScalarKind::kUnsupported;
  MemRefInfo memref;
};

ScalarKind classifyScalar(mlir::Type type);
TypeInfo classifyType(mlir::Type type);
void ensureSupported(const TypeInfo &info, const std::string &label);
void ensureStaticDim(int64_t expected, int64_t actual, const std::string &label);
SEXP ensureRType(SEXP sexp, SEXPTYPE type, bool expectMatrix, const std::string &label);
void copyColToRow(const Rcpp::NumericMatrix &mat, std::vector<double> &out);
void copyColToRow(const Rcpp::IntegerMatrix &mat, std::vector<int32_t> &out);
void copyRowToCol(const std::vector<double> &row, int nrow, int ncol, Rcpp::NumericMatrix &out);
void copyRowToCol(const std::vector<int32_t> &row, int nrow, int ncol, Rcpp::IntegerMatrix &out);

struct ArgBase {
  virtual ~ArgBase() = default;
  virtual void *addr() = 0;
  virtual void copyBack() {}
};

struct ScalarI32Arg : ArgBase {
  int32_t value;
  explicit ScalarI32Arg(SEXP sexp);
  void *addr() override;
};

struct ScalarF64Arg : ArgBase {
  double value;
  explicit ScalarF64Arg(SEXP sexp);
  void *addr() override;
};

struct ScalarBoolArg : ArgBase {
  bool value;
  explicit ScalarBoolArg(SEXP sexp);
  void *addr() override;
};

struct ScalarIndexArg : ArgBase {
  int64_t value;
  explicit ScalarIndexArg(SEXP sexp);
  void *addr() override;
};

struct VectorI32Arg : ArgBase {
  Rcpp::IntegerVector vec;
  StridedMemRefType<int32_t, 1> desc;
  void *descriptorPtr = nullptr;

  VectorI32Arg(SEXP sexp, const MemRefInfo &info);
  void *addr() override;
};

struct VectorF64Arg : ArgBase {
  Rcpp::NumericVector vec;
  StridedMemRefType<double, 1> desc;
  void *descriptorPtr = nullptr;

  VectorF64Arg(SEXP sexp, const MemRefInfo &info);
  void *addr() override;
};

struct MatrixI32Arg : ArgBase {
  Rcpp::IntegerMatrix mat;
  std::vector<int32_t> scratch;
  StridedMemRefType<int32_t, 2> desc;
  void *descriptorPtr = nullptr;

  MatrixI32Arg(SEXP sexp, const MemRefInfo &info);
  void copyBack() override;
  void *addr() override;
};

struct MatrixF64Arg : ArgBase {
  Rcpp::NumericMatrix mat;
  std::vector<double> scratch;
  StridedMemRefType<double, 2> desc;
  void *descriptorPtr = nullptr;

  MatrixF64Arg(SEXP sexp, const MemRefInfo &info);
  void copyBack() override;
  void *addr() override;
};

} // namespace rdslmlir

#endif // RDSLMLIR_RUNTIME_RUNTIME_TYPES_H

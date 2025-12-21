#ifndef RDSLMLIR_RUNTIME_TYPES_ARGS_H
#define RDSLMLIR_RUNTIME_TYPES_ARGS_H

#include <Rcpp.h>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "rdslmlir/Runtime/Types/TypeInfo.h"

#include <cstdint>
#include <vector>

namespace rdslmlir {

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

#endif // RDSLMLIR_RUNTIME_TYPES_ARGS_H

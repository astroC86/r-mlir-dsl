#include "rdslmlir/Runtime/RuntimeTypes.h"

using namespace mlir;

namespace rdslmlir {

ScalarKind classifyScalar(Type type) {
  if (type.isIndex()) return ScalarKind::kIndex;
  if (type.isInteger(1)) return ScalarKind::kBool;
  if (type.isInteger(32)) return ScalarKind::kI32;
  if (type.isF64()) return ScalarKind::kF64;
  return ScalarKind::kUnsupported;
}

TypeInfo classifyType(Type type) {
  TypeInfo info;
  if (auto memref = llvm::dyn_cast<MemRefType>(type)) {
    info.isMemRef = true;
    info.memref.rank = memref.getRank();
    info.memref.elem = classifyScalar(memref.getElementType());
    info.memref.shape.assign(memref.getShape().begin(), memref.getShape().end());
    return info;
  }
  info.scalar = classifyScalar(type);
  return info;
}

void ensureSupported(const TypeInfo &info, const std::string &label) {
  if (info.isMemRef) {
    if (info.memref.rank < 1 || info.memref.rank > 2) {
      Rcpp::stop(label + " only supports rank-1/2 memrefs");
    }
    if (info.memref.elem != ScalarKind::kI32 && info.memref.elem != ScalarKind::kF64) {
      Rcpp::stop(label + " only supports i32/f64 memrefs");
    }
    return;
  }
  if (info.scalar == ScalarKind::kUnsupported) {
    Rcpp::stop(label + " only supports i32/f64/bool/index scalars");
  }
}

void ensureStaticDim(int64_t expected, int64_t actual, const std::string &label) {
  if (expected == ShapedType::kDynamic) return;
  if (expected != actual) {
    Rcpp::stop(label + " expects size " + std::to_string(expected) +
               ", got " + std::to_string(actual));
  }
}

SEXP ensureRType(SEXP sexp, SEXPTYPE type, bool expectMatrix, const std::string &label) {
  if (TYPEOF(sexp) != static_cast<int>(type)) Rcpp::stop("expected " + label);
  if (expectMatrix && !Rf_isMatrix(sexp)) Rcpp::stop("expected " + label);
  if (!expectMatrix && Rf_isMatrix(sexp)) Rcpp::stop("expected " + label);
  return sexp;
}

void copyColToRow(const Rcpp::NumericMatrix &mat, std::vector<double> &out) {
  int nrow = mat.nrow();
  int ncol = mat.ncol();
  out.resize(static_cast<size_t>(nrow) * static_cast<size_t>(ncol));
  for (int j = 0; j < ncol; ++j) {
    for (int i = 0; i < nrow; ++i) {
      out[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)] = mat(i, j);
    }
  }
}

void copyColToRow(const Rcpp::IntegerMatrix &mat, std::vector<int32_t> &out) {
  int nrow = mat.nrow();
  int ncol = mat.ncol();
  out.resize(static_cast<size_t>(nrow) * static_cast<size_t>(ncol));
  for (int j = 0; j < ncol; ++j) {
    for (int i = 0; i < nrow; ++i) {
      out[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)] = mat(i, j);
    }
  }
}

void copyRowToCol(const std::vector<double> &row, int nrow, int ncol, Rcpp::NumericMatrix &out) {
  for (int j = 0; j < ncol; ++j) {
    for (int i = 0; i < nrow; ++i) {
      out(i, j) = row[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)];
    }
  }
}

void copyRowToCol(const std::vector<int32_t> &row, int nrow, int ncol, Rcpp::IntegerMatrix &out) {
  for (int j = 0; j < ncol; ++j) {
    for (int i = 0; i < nrow; ++i) {
      out(i, j) = row[static_cast<size_t>(i) * static_cast<size_t>(ncol) + static_cast<size_t>(j)];
    }
  }
}

ScalarI32Arg::ScalarI32Arg(SEXP sexp) : value(Rcpp::as<int32_t>(sexp)) {}
void *ScalarI32Arg::addr() { return &value; }

ScalarF64Arg::ScalarF64Arg(SEXP sexp) : value(Rcpp::as<double>(sexp)) {}
void *ScalarF64Arg::addr() { return &value; }

ScalarBoolArg::ScalarBoolArg(SEXP sexp) : value(Rcpp::as<bool>(sexp)) {}
void *ScalarBoolArg::addr() { return &value; }

ScalarIndexArg::ScalarIndexArg(SEXP sexp) : value(Rcpp::as<int64_t>(sexp)) {}
void *ScalarIndexArg::addr() { return &value; }

VectorI32Arg::VectorI32Arg(SEXP sexp, const MemRefInfo &info)
    : vec(ensureRType(sexp, INTSXP, false, "integer vector")) {
  int64_t n = vec.size();
  ensureStaticDim(info.shape[0], n, "vector parameter");
  desc.basePtr = vec.begin();
  desc.data = vec.begin();
  desc.offset = 0;
  desc.sizes[0] = n;
  desc.strides[0] = 1;
  descriptorPtr = &desc;
}

void *VectorI32Arg::addr() { return &descriptorPtr; }

VectorF64Arg::VectorF64Arg(SEXP sexp, const MemRefInfo &info)
    : vec(ensureRType(sexp, REALSXP, false, "numeric vector")) {
  int64_t n = vec.size();
  ensureStaticDim(info.shape[0], n, "vector parameter");
  desc.basePtr = vec.begin();
  desc.data = vec.begin();
  desc.offset = 0;
  desc.sizes[0] = n;
  desc.strides[0] = 1;
  descriptorPtr = &desc;
}

void *VectorF64Arg::addr() { return &descriptorPtr; }

MatrixI32Arg::MatrixI32Arg(SEXP sexp, const MemRefInfo &info)
    : mat(ensureRType(sexp, INTSXP, true, "integer matrix")) {
  int64_t nrow = mat.nrow();
  int64_t ncol = mat.ncol();
  ensureStaticDim(info.shape[0], nrow, "matrix parameter");
  ensureStaticDim(info.shape[1], ncol, "matrix parameter");
  copyColToRow(mat, scratch);
  desc.basePtr = scratch.data();
  desc.data = scratch.data();
  desc.offset = 0;
  desc.sizes[0] = nrow;
  desc.sizes[1] = ncol;
  desc.strides[0] = ncol;
  desc.strides[1] = 1;
  descriptorPtr = &desc;
}

void MatrixI32Arg::copyBack() {
  copyRowToCol(scratch, mat.nrow(), mat.ncol(), mat);
}

void *MatrixI32Arg::addr() { return &descriptorPtr; }

MatrixF64Arg::MatrixF64Arg(SEXP sexp, const MemRefInfo &info)
    : mat(ensureRType(sexp, REALSXP, true, "numeric matrix")) {
  int64_t nrow = mat.nrow();
  int64_t ncol = mat.ncol();
  ensureStaticDim(info.shape[0], nrow, "matrix parameter");
  ensureStaticDim(info.shape[1], ncol, "matrix parameter");
  copyColToRow(mat, scratch);
  desc.basePtr = scratch.data();
  desc.data = scratch.data();
  desc.offset = 0;
  desc.sizes[0] = nrow;
  desc.sizes[1] = ncol;
  desc.strides[0] = ncol;
  desc.strides[1] = 1;
  descriptorPtr = &desc;
}

void MatrixF64Arg::copyBack() {
  copyRowToCol(scratch, mat.nrow(), mat.ncol(), mat);
}

void *MatrixF64Arg::addr() { return &descriptorPtr; }

} // namespace rdslmlir

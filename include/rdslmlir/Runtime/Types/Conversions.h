#ifndef RDSLMLIR_RUNTIME_TYPES_CONVERSIONS_H
#define RDSLMLIR_RUNTIME_TYPES_CONVERSIONS_H

#include <Rcpp.h>

#include <cstdint>
#include <string>
#include <vector>

namespace rdslmlir {

SEXP ensureRType(SEXP sexp, SEXPTYPE type, bool expectMatrix, const std::string &label);

void copyColToRow(const Rcpp::NumericMatrix &mat, std::vector<double> &out);
void copyColToRow(const Rcpp::IntegerMatrix &mat, std::vector<int32_t> &out);
void copyRowToCol(const std::vector<double> &row, int nrow, int ncol, Rcpp::NumericMatrix &out);
void copyRowToCol(const std::vector<int32_t> &row, int nrow, int ncol, Rcpp::IntegerMatrix &out);

} // namespace rdslmlir

#endif // RDSLMLIR_RUNTIME_TYPES_CONVERSIONS_H

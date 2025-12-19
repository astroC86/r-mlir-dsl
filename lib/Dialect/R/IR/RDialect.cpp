#include "rdslmlir/Dialect/R/IR/RDialect.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace r;

#include "rdslmlir/Dialect/R/IR/RDialect.cpp.inc"

void RDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "rdslmlir/Dialect/R/IR/ROps.cpp.inc"
      >();
}

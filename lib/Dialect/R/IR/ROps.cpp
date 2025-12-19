#include "rdslmlir/Dialect/R/IR/ROps.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace r;

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

#define GET_OP_CLASSES
#include "rdslmlir/Dialect/R/IR/ROps.cpp.inc"

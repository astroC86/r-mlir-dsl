#ifndef RDSLMLIR_DIALECT_R_IR_R_OPS_H
#define RDSLMLIR_DIALECT_R_IR_R_OPS_H

#include "rdslmlir/Dialect/R/IR/RDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "rdslmlir/Dialect/R/IR/ROps.h.inc"

#endif // RDSLMLIR_DIALECT_R_IR_R_OPS_H

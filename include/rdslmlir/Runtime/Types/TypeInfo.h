#ifndef RDSLMLIR_RUNTIME_TYPES_TYPE_INFO_H
#define RDSLMLIR_RUNTIME_TYPES_TYPE_INFO_H

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

} // namespace rdslmlir

#endif // RDSLMLIR_RUNTIME_TYPES_TYPE_INFO_H

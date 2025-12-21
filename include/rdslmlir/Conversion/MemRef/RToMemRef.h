#ifndef RDSLMLIR_CONVERSION_MEMREF_R_TO_MEMREF_H
#define RDSLMLIR_CONVERSION_MEMREF_R_TO_MEMREF_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToMemRefPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_MEMREF_R_TO_MEMREF_H

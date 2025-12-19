#ifndef RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_PATTERNS_H
#define RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToSCFPatterns(mlir::RewritePatternSet &patterns);
void populateRLowerToLinalgPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_R_TO_LINALG_LOWERING_PATTERNS_H

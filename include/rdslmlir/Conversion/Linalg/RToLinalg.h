#ifndef RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_H
#define RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToLinalgPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_H

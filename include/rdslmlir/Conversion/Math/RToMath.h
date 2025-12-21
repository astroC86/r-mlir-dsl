#ifndef RDSLMLIR_CONVERSION_MATH_R_TO_MATH_H
#define RDSLMLIR_CONVERSION_MATH_R_TO_MATH_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToMathSCFPatterns(mlir::RewritePatternSet &patterns);
void populateRLowerToMathLinalgPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_MATH_R_TO_MATH_H

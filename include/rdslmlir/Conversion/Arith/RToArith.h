#ifndef RDSLMLIR_CONVERSION_ARITH_R_TO_ARITH_H
#define RDSLMLIR_CONVERSION_ARITH_R_TO_ARITH_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToArithSCFPatterns(mlir::RewritePatternSet &patterns);
void populateRLowerToArithLinalgPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_ARITH_R_TO_ARITH_H

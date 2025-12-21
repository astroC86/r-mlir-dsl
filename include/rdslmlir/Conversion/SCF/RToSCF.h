#ifndef RDSLMLIR_CONVERSION_SCF_R_TO_SCF_H
#define RDSLMLIR_CONVERSION_SCF_R_TO_SCF_H

#include "mlir/IR/PatternMatch.h"

namespace r {

void populateRLowerToSCFPatterns(mlir::RewritePatternSet &patterns);

} // namespace r

#endif // RDSLMLIR_CONVERSION_SCF_R_TO_SCF_H

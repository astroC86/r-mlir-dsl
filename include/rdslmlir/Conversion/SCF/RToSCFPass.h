#ifndef RDSLMLIR_CONVERSION_SCF_R_TO_SCF_PASS_H
#define RDSLMLIR_CONVERSION_SCF_R_TO_SCF_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace r {

std::unique_ptr<mlir::Pass> createRLowerToSCFPass();

} // namespace r

#endif // RDSLMLIR_CONVERSION_SCF_R_TO_SCF_PASS_H

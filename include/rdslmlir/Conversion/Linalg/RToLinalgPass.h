#ifndef RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_PASS_H
#define RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_PASS_H

#include <memory>

namespace mlir {
class Pass;
}

namespace r {

std::unique_ptr<mlir::Pass> createRLowerToLinalgPass();

} // namespace r

#endif // RDSLMLIR_CONVERSION_LINALG_R_TO_LINALG_PASS_H

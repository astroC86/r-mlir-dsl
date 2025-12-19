#ifndef RDSLMLIR_CONVERSION_RPASSES_H
#define RDSLMLIR_CONVERSION_RPASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace r {
std::unique_ptr<mlir::Pass> createRLowerToSCFPass();
std::unique_ptr<mlir::Pass> createRLowerToLinalgPass();
void registerRPasses();
} // namespace r

#endif // RDSLMLIR_CONVERSION_RPASSES_H

#include "rdslmlir/Conversion/OpenMP/RToOpenMP.h"

#include "rdslmlir/Conversion/Linalg/RToLinalgPass.h"

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void r::registerRToOpenMPPipelines() {
  PassPipelineRegistration<>(
      "r-to-openmp",
      "Lower R dialect and map parallel loops to OpenMP",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
        pm.addPass(mlir::createConvertSCFToOpenMPPass());
      });
}

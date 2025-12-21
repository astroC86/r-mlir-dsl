#include "rdslmlir/Conversion/RegisterAll.h"

#include "rdslmlir/Conversion/Linalg/RToLinalgPass.h"
#include "rdslmlir/Conversion/OpenMP/RToOpenMP.h"
#include "rdslmlir/Conversion/SCF/RToSCFPass.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void r::registerRPasses() {
  registerPass(createRLowerToSCFPass);
  registerPass(createRLowerToLinalgPass);

  PassPipelineRegistration<>(
      "r-to-scf",
      "Lower R dialect to SCF/arith/memref",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
      });

  PassPipelineRegistration<>(
      "r-to-linalg",
      "Lower R dialect to Linalg/memref",
      [](OpPassManager &pm) { pm.addPass(createRLowerToLinalgPass()); });

  registerRToOpenMPPipelines();

  PassPipelineRegistration<>(
      "r-to-gpu",
      "Lower R dialect and map parallel loops to GPU",
      [](OpPassManager &pm) {
        pm.addPass(createRLowerToLinalgPass());
        pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
        pm.addPass(mlir::createConvertParallelLoopToGpuPass());
      });
}

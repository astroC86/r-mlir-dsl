#include "rdslmlir/Conversion/RegisterAll.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<r::RDialect>();
  mlir::registerAllDialects(registry);
  r::registerRPasses();
  mlir::registerAllPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "R MLIR optimizer\n", registry));
}

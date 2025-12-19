add_mlir_library(RConversion
  LoweringUtils.cpp
  LoweringPatterns.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  RIR
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFToGPU
  MLIRSCFToOpenMP
  MLIRTransforms
)

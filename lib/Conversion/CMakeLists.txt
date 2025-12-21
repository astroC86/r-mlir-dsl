add_mlir_library(RConversion
  Arith/RToArith.cpp
  Common/LoweringUtils.cpp
  Linalg/RToLinalg.cpp
  Linalg/RToLinalgPass.cpp
  Math/RToMath.cpp
  MemRef/RToMemRef.cpp
  OpenMP/RToOpenMP.cpp
  SCF/RToSCF.cpp
  SCF/RToSCFPass.cpp
  RegisterAll.cpp

  LINK_LIBS PUBLIC
  RIR
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFToGPU
  MLIRSCFToOpenMP
  MLIRTransforms
)

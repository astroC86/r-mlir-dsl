add_mlir_tool(r-ast-to-mlir
  r-ast-to-mlir.cpp
)

target_link_libraries(r-ast-to-mlir PRIVATE
  RIR
  MLIRBufferizationDialect
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRSupport
)

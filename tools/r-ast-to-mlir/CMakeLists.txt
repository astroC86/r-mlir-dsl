add_mlir_tool(r-ast-to-mlir
  r-ast-to-mlir.cpp
)

target_link_libraries(r-ast-to-mlir PRIVATE
  rdslmlir_ast_lowerer
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

add_test(NAME r_ast_to_mlir_smoke
  COMMAND $<TARGET_FILE:r-ast-to-mlir>
  ${CMAKE_CURRENT_SOURCE_DIR}/testdata/basic.json
)

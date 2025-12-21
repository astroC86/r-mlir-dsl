add_library(rdslmlir_ast_lowerer STATIC
  RuntimeAstLowerer.cpp
  RuntimeAstLowererExpr.cpp
  RuntimeAstLowererStmt.cpp
)

target_include_directories(rdslmlir_ast_lowerer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(rdslmlir_ast_lowerer PRIVATE
  RIR
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)

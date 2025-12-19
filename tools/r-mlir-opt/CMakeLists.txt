add_mlir_tool(r-mlir-opt
  r-mlir-opt.cpp
)

target_link_libraries(r-mlir-opt PRIVATE
  RIR
  RConversion
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRTransforms
)

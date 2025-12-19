add_mlir_library(RIR
  RDialect.cpp
  ROps.cpp

  DEPENDS
  RTableGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(RIR PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)

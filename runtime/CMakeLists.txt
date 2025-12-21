set(RDSLMLIR_RUNTIME_SOURCES
  Execution/rdslmlir_runtime.cpp
  Runtime/RuntimeTypes.cpp
)

add_library(rdslmlir_runtime SHARED ${RDSLMLIR_RUNTIME_SOURCES})

# R and Rcpp flags
execute_process(
  COMMAND R CMD config --cppflags
  OUTPUT_VARIABLE RDSLMLIR_R_CPPFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND R CMD config --ldflags
  OUTPUT_VARIABLE RDSLMLIR_R_LDFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND Rscript -e "cat(Rcpp:::CxxFlags())"
  OUTPUT_VARIABLE RDSLMLIR_RCPP_CPPFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND Rscript -e "cat(Rcpp:::LdFlags())"
  OUTPUT_VARIABLE RDSLMLIR_RCPP_LDFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(RDSLMLIR_R_CPPFLAGS UNIX_COMMAND "${RDSLMLIR_R_CPPFLAGS}")
separate_arguments(RDSLMLIR_R_LDFLAGS UNIX_COMMAND "${RDSLMLIR_R_LDFLAGS}")
separate_arguments(RDSLMLIR_RCPP_CPPFLAGS UNIX_COMMAND "${RDSLMLIR_RCPP_CPPFLAGS}")
separate_arguments(RDSLMLIR_RCPP_LDFLAGS UNIX_COMMAND "${RDSLMLIR_RCPP_LDFLAGS}")

find_library(RDSLMLIR_TIRPC_LIB NAMES tirpc)
if (NOT RDSLMLIR_TIRPC_LIB)
  list(REMOVE_ITEM RDSLMLIR_R_LDFLAGS "-ltirpc")
  list(REMOVE_ITEM RDSLMLIR_RCPP_LDFLAGS "-ltirpc")
endif()

if (LLVM_LIBRARY_DIRS)
  list(GET LLVM_LIBRARY_DIRS 0 RDSLMLIR_LLVM_LIB_DIR)
  target_compile_definitions(rdslmlir_runtime PRIVATE "RDSLMLIR_LLVM_LIB_DIR=\"${RDSLMLIR_LLVM_LIB_DIR}\"")
endif()

if (LLVM_LIBRARY_DIRS)
  find_library(RDSLMLIR_C_RUNNER_UTILS NAMES mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIRS})
else()
  find_library(RDSLMLIR_C_RUNNER_UTILS NAMES mlir_c_runner_utils)
endif()
if (RDSLMLIR_C_RUNNER_UTILS)
  target_link_libraries(rdslmlir_runtime PRIVATE ${RDSLMLIR_C_RUNNER_UTILS})
endif()

if (RDSLMLIR_R_CPPFLAGS)
  target_compile_options(rdslmlir_runtime PRIVATE ${RDSLMLIR_R_CPPFLAGS})
endif()
if (RDSLMLIR_RCPP_CPPFLAGS)
  target_compile_options(rdslmlir_runtime PRIVATE ${RDSLMLIR_RCPP_CPPFLAGS})
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(rdslmlir_runtime PRIVATE -Wno-cast-function-type -Wno-suggest-override)
endif()

set(RDSLMLIR_LINK_DIRS)
set(RDSLMLIR_LINK_LIBS)
set(RDSLMLIR_LINK_OPTIONS)

foreach(flag IN LISTS RDSLMLIR_R_LDFLAGS RDSLMLIR_RCPP_LDFLAGS)
  if (flag MATCHES "^-L(.+)")
    list(APPEND RDSLMLIR_LINK_DIRS "${CMAKE_MATCH_1}")
  elseif (flag MATCHES "^-l(.+)")
    list(APPEND RDSLMLIR_LINK_LIBS "${CMAKE_MATCH_1}")
  else()
    list(APPEND RDSLMLIR_LINK_OPTIONS "${flag}")
  endif()
endforeach()

if (RDSLMLIR_LINK_DIRS)
  target_link_directories(rdslmlir_runtime PRIVATE ${RDSLMLIR_LINK_DIRS})
endif()
if (RDSLMLIR_LINK_OPTIONS)
  target_link_options(rdslmlir_runtime PRIVATE ${RDSLMLIR_LINK_OPTIONS})
endif()
if (RDSLMLIR_LINK_LIBS)
  target_link_libraries(rdslmlir_runtime PRIVATE ${RDSLMLIR_LINK_LIBS})
endif()

target_include_directories(rdslmlir_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(rdslmlir_runtime PRIVATE
  rdslmlir_ast_lowerer
  RConversion
  RIR
  MLIRExecutionEngine
  MLIRExecutionEngineUtils
  MLIRMathToLLVM
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRIR
  MLIROpenMPToLLVM
  MLIRParser
  MLIRPass
  MLIRSCFToOpenMP
  MLIRTransforms
  MLIRSupport
)

set_target_properties(rdslmlir_runtime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/runtime
)

if (LLVM_LIBRARY_DIRS)
  set_target_properties(rdslmlir_runtime PROPERTIES
    BUILD_RPATH "${LLVM_LIBRARY_DIRS}"
  )
endif()

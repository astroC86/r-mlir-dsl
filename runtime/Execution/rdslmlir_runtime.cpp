#include <Rcpp.h>

#include "rdslmlir/Conversion/Linalg/RToLinalgPass.h"
#include "rdslmlir/Conversion/RegisterAll.h"
#include "rdslmlir/Dialect/R/IR/RDialect.h"
#include "rdslmlir/Dialect/R/IR/ROps.h"
#include "rdslmlir/Runtime/RuntimeAstLowerer.h"
#include "rdslmlir/Runtime/RuntimeTypes.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

extern "C" void mlirFree(void *ptr);

namespace {
using namespace rdslmlir;

static bool shouldEnableOpenMP(const llvm::json::Value &value) {
  if (auto obj = value.getAsObject()) {
    if (auto target = obj->getString("target")) {
      if (*target == "openmp") return true;
    }
    if (auto type = obj->getString("type")) {
      if (*type == "For") {
        bool parallel = false;
        if (auto p = obj->getBoolean("parallel")) parallel = *p;
        if (parallel) {
          auto target = obj->getString("target");
          if (!target || *target == "openmp" || *target == "cpu") return true;
        }
      }
    }
    for (auto &entry : *obj) {
      if (shouldEnableOpenMP(entry.second)) return true;
    }
    return false;
  }
  if (auto arr = value.getAsArray()) {
    for (auto &item : *arr) {
      if (shouldEnableOpenMP(item)) return true;
    }
  }
  return false;
}

static bool hasOpenMPSymbols() {
  return llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__kmpc_fork_call") &&
         llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__kmpc_global_thread_num");
}

static void loadOpenMPRuntime() {
  if (hasOpenMPSymbols()) return;

  std::vector<std::string> candidates;
  if (const char *path = std::getenv("RDSLMLIR_OMP_LIB")) {
    candidates.emplace_back(path);
  } else {
    #ifdef RDSLMLIR_LLVM_LIB_DIR
    candidates.emplace_back(std::string(RDSLMLIR_LLVM_LIB_DIR) + "/libomp.so");
    candidates.emplace_back(std::string(RDSLMLIR_LLVM_LIB_DIR) + "/libomp.so.5");
    #endif
    candidates.emplace_back("libomp.so");
    candidates.emplace_back("libomp.so.5");
    candidates.emplace_back("libgomp.so.1");
  }

  std::string errMsg;
  for (const auto &lib : candidates) {
    errMsg.clear();
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(lib.c_str(), &errMsg)) {
      continue;
    }
    if (hasOpenMPSymbols()) return;
  }

  Rcpp::stop("failed to load OpenMP runtime (__kmpc symbols not found); set RDSLMLIR_OMP_LIB to libomp");
}

struct RuntimeHandle {
  std::string entry;
  std::vector<TypeInfo> params;
  TypeInfo result;
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  std::unique_ptr<ExecutionEngine> engine;
};

static std::unique_ptr<RuntimeHandle> compileModule(const std::string &json,
                                                    const std::string &entry) {
  static bool initialized = false;
  static bool passesRegistered = false;
  if (!initialized) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
  }
  if (!passesRegistered) {
    r::registerRPasses();
    registerAllPasses();
    passesRegistered = true;
  }

  auto parsed = llvm::json::parse(json);
  if (!parsed) {
    Rcpp::stop("invalid JSON for module");
  }
  bool enableOpenMP = shouldEnableOpenMP(*parsed);
  auto *root = parsed->getAsObject();
  if (!root) {
    Rcpp::stop("module JSON must be an object");
  }

  DialectRegistry registry;
  registry.insert<r::RDialect, func::FuncDialect, arith::ArithDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect, linalg::LinalgDialect,
                  cf::ControlFlowDialect, omp::OpenMPDialect>();
  registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerOpenMPDialectTranslation(registry);

  auto ctx = std::make_unique<MLIRContext>(registry);
  ctx->loadAllAvailableDialects();

  OpBuilder builder(ctx.get());
  OwningOpRef<ModuleOp> module = ModuleOp::create(builder.getUnknownLoc());

  AstLowerer lowerer(*ctx, *module);
  if (failed(lowerer.lowerModule(*root))) {
    Rcpp::stop("failed to lower JSON AST to MLIR");
  }

  if (failed(verify(*module))) {
    Rcpp::stop("MLIR module verification failed");
  }

  auto func = module->lookupSymbol<func::FuncOp>(entry);
  if (!func) {
    Rcpp::stop("entry function not found: " + entry);
  }
  func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                UnitAttr::get(func.getContext()));

  RuntimeHandle handle;
  handle.entry = entry;
  handle.context = std::move(ctx);
  handle.module = std::move(module);

  auto funcType = func.getFunctionType();
  handle.params.reserve(funcType.getNumInputs());
  for (Type ty : funcType.getInputs()) {
    TypeInfo info = classifyType(ty);
    ensureSupported(info, "parameter type");
    handle.params.push_back(info);
  }

  if (funcType.getNumResults() == 0) {
    handle.result.isMemRef = false;
    handle.result.scalar = ScalarKind::kUnsupported;
  } else if (funcType.getNumResults() == 1) {
    handle.result = classifyType(funcType.getResult(0));
    ensureSupported(handle.result, "return type");
  } else {
    Rcpp::stop("entry function must have at most one result");
  }

  PassManager pm(handle.context.get());
  pm.addPass(r::createRLowerToLinalgPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertBufferizationToMemRefPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  if (enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
  }
  pm.addPass(createSCFToControlFlowPass());
  if (enableOpenMP) {
    pm.addPass(createConvertOpenMPToLLVMPass());
  }
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(*handle.module))) {
    Rcpp::stop("failed to lower MLIR to LLVM");
  }

  if (enableOpenMP) {
    loadOpenMPRuntime();
  }

  ExecutionEngineOptions options;
#ifdef RDSLMLIR_LLVM_LIB_DIR
  std::string libDir = RDSLMLIR_LLVM_LIB_DIR;
  std::string cRunner = libDir + "/libmlir_c_runner_utils.so";
  std::string runner = libDir + "/libmlir_runner_utils.so";
  options.sharedLibPaths = {cRunner, runner};
#endif

  auto engineExpected = ExecutionEngine::create(*handle.module, options);
  if (!engineExpected) {
    std::string err = llvm::toString(engineExpected.takeError());
    Rcpp::stop("failed to create execution engine: " + err);
  }
  handle.engine = std::move(engineExpected.get());

  return std::unique_ptr<RuntimeHandle>(new RuntimeHandle(std::move(handle)));
}

static Rcpp::RObject buildScalarResult(const TypeInfo &info, void *resultPtr) {
  switch (info.scalar) {
    case ScalarKind::kI32:
      return Rcpp::wrap(*reinterpret_cast<int32_t *>(resultPtr));
    case ScalarKind::kF64:
      return Rcpp::wrap(*reinterpret_cast<double *>(resultPtr));
    case ScalarKind::kBool:
      return Rcpp::wrap(*reinterpret_cast<bool *>(resultPtr));
    case ScalarKind::kIndex:
      return Rcpp::wrap(static_cast<double>(*reinterpret_cast<int64_t *>(resultPtr)));
    default:
      break;
  }
  return R_NilValue;
}

} // namespace

extern "C" SEXP rdslmlir_compile(SEXP json, SEXP entry) {
  std::string jsonStr = Rcpp::as<std::string>(json);
  std::string entryStr = Rcpp::as<std::string>(entry);
  auto handle = compileModule(jsonStr, entryStr);
  Rcpp::XPtr<RuntimeHandle> ptr(handle.release(), true);
  return ptr;
}

extern "C" SEXP rdslmlir_call(SEXP handle, SEXP args) {
  Rcpp::XPtr<RuntimeHandle> ptr(handle);
  if (!ptr) Rcpp::stop("invalid runtime handle");
  if (TYPEOF(args) != VECSXP) Rcpp::stop("args must be a list");

  Rcpp::List argList(args);
  if (argList.size() != static_cast<int>(ptr->params.size())) {
    Rcpp::stop("argument count does not match function parameters");
  }

  std::vector<std::unique_ptr<ArgBase>> ownedArgs;
  ownedArgs.reserve(ptr->params.size());
  llvm::SmallVector<void *, 8> packed;

  for (int i = 0; i < argList.size(); ++i) {
    const TypeInfo &info = ptr->params[static_cast<size_t>(i)];
    SEXP arg = argList[i];

    if (!info.isMemRef) {
      switch (info.scalar) {
        case ScalarKind::kI32:
          ownedArgs.emplace_back(new ScalarI32Arg(arg));
          break;
        case ScalarKind::kF64:
          ownedArgs.emplace_back(new ScalarF64Arg(arg));
          break;
        case ScalarKind::kBool:
          ownedArgs.emplace_back(new ScalarBoolArg(arg));
          break;
        case ScalarKind::kIndex:
          ownedArgs.emplace_back(new ScalarIndexArg(arg));
          break;
        default:
          Rcpp::stop("unsupported scalar parameter type");
      }
      packed.push_back(ownedArgs.back()->addr());
      continue;
    }

    const MemRefInfo &memref = info.memref;
    if (memref.rank == 1) {
      if (memref.elem == ScalarKind::kI32) {
        ownedArgs.emplace_back(new VectorI32Arg(arg, memref));
      } else if (memref.elem == ScalarKind::kF64) {
        ownedArgs.emplace_back(new VectorF64Arg(arg, memref));
      } else {
        Rcpp::stop("unsupported vector element type");
      }
      packed.push_back(ownedArgs.back()->addr());
      continue;
    }

    if (memref.rank == 2) {
      if (memref.elem == ScalarKind::kI32) {
        ownedArgs.emplace_back(new MatrixI32Arg(arg, memref));
      } else if (memref.elem == ScalarKind::kF64) {
        ownedArgs.emplace_back(new MatrixF64Arg(arg, memref));
      } else {
        Rcpp::stop("unsupported matrix element type");
      }
      packed.push_back(ownedArgs.back()->addr());
      continue;
    }

    Rcpp::stop("unsupported memref rank");
  }

  std::string cifaceName = std::string("_mlir_ciface_") + ptr->entry;
  if (!ptr->result.isMemRef && ptr->result.scalar == ScalarKind::kUnsupported) {
    auto err = ptr->engine->invokePacked(cifaceName, packed);
    if (err) {
      std::string msg = llvm::toString(std::move(err));
      Rcpp::stop("runtime invoke failed: " + msg);
    }
    for (auto &arg : ownedArgs) arg->copyBack();
    return R_NilValue;
  }

  if (!ptr->result.isMemRef) {
    int32_t i32Result = 0;
    double f64Result = 0.0;
    bool boolResult = false;
    int64_t indexResult = 0;
    void *resultPtr = nullptr;

    switch (ptr->result.scalar) {
      case ScalarKind::kI32:
        resultPtr = &i32Result;
        break;
      case ScalarKind::kF64:
        resultPtr = &f64Result;
        break;
      case ScalarKind::kBool:
        resultPtr = &boolResult;
        break;
      case ScalarKind::kIndex:
        resultPtr = &indexResult;
        break;
      default:
        Rcpp::stop("unsupported scalar return type");
    }

    packed.push_back(resultPtr);
    auto err = ptr->engine->invokePacked(cifaceName, packed);
    if (err) {
      std::string msg = llvm::toString(std::move(err));
      Rcpp::stop("runtime invoke failed: " + msg);
    }

    for (auto &arg : ownedArgs) arg->copyBack();
    return buildScalarResult(ptr->result, resultPtr);
  }

  const MemRefInfo &ret = ptr->result.memref;
  if (ret.rank == 1) {
    if (ret.elem == ScalarKind::kI32) {
      StridedMemRefType<int32_t, 1> out = {};
      void *outPtr = &out;
      llvm::SmallVector<void *, 8> callArgs;
      callArgs.reserve(packed.size() + 1);
      callArgs.push_back(&outPtr);
      callArgs.append(packed.begin(), packed.end());
      auto err = ptr->engine->invokePacked(cifaceName, callArgs);
      if (err) {
        std::string msg = llvm::toString(std::move(err));
        Rcpp::stop("runtime invoke failed: " + msg);
      }
      for (auto &arg : ownedArgs) arg->copyBack();
      int64_t n = out.sizes[0];
      Rcpp::IntegerVector res(n);
      for (int64_t i = 0; i < n; ++i) {
        res[i] = out.data[out.offset + i * out.strides[0]];
      }
      mlirFree(out.basePtr);
      return res;
    }
    if (ret.elem == ScalarKind::kF64) {
      StridedMemRefType<double, 1> out = {};
      void *outPtr = &out;
      llvm::SmallVector<void *, 8> callArgs;
      callArgs.reserve(packed.size() + 1);
      callArgs.push_back(&outPtr);
      callArgs.append(packed.begin(), packed.end());
      auto err = ptr->engine->invokePacked(cifaceName, callArgs);
      if (err) {
        std::string msg = llvm::toString(std::move(err));
        Rcpp::stop("runtime invoke failed: " + msg);
      }
      for (auto &arg : ownedArgs) arg->copyBack();
      int64_t n = out.sizes[0];
      Rcpp::NumericVector res(n);
      for (int64_t i = 0; i < n; ++i) {
        res[i] = out.data[out.offset + i * out.strides[0]];
      }
      mlirFree(out.basePtr);
      return res;
    }
    Rcpp::stop("unsupported vector return type");
  }

  if (ret.rank == 2) {
    if (ret.elem == ScalarKind::kI32) {
      StridedMemRefType<int32_t, 2> out = {};
      void *outPtr = &out;
      llvm::SmallVector<void *, 8> callArgs;
      callArgs.reserve(packed.size() + 1);
      callArgs.push_back(&outPtr);
      callArgs.append(packed.begin(), packed.end());
      auto err = ptr->engine->invokePacked(cifaceName, callArgs);
      if (err) {
        std::string msg = llvm::toString(std::move(err));
        Rcpp::stop("runtime invoke failed: " + msg);
      }
      for (auto &arg : ownedArgs) arg->copyBack();
      int64_t nrow = out.sizes[0];
      int64_t ncol = out.sizes[1];
      Rcpp::IntegerMatrix res(nrow, ncol);
      for (int64_t j = 0; j < ncol; ++j) {
        for (int64_t i = 0; i < nrow; ++i) {
          res(i, j) = out.data[out.offset + i * out.strides[0] + j * out.strides[1]];
        }
      }
      mlirFree(out.basePtr);
      return res;
    }
    if (ret.elem == ScalarKind::kF64) {
      StridedMemRefType<double, 2> out = {};
      void *outPtr = &out;
      llvm::SmallVector<void *, 8> callArgs;
      callArgs.reserve(packed.size() + 1);
      callArgs.push_back(&outPtr);
      callArgs.append(packed.begin(), packed.end());
      auto err = ptr->engine->invokePacked(cifaceName, callArgs);
      if (err) {
        std::string msg = llvm::toString(std::move(err));
        Rcpp::stop("runtime invoke failed: " + msg);
      }
      for (auto &arg : ownedArgs) arg->copyBack();
      int64_t nrow = out.sizes[0];
      int64_t ncol = out.sizes[1];
      Rcpp::NumericMatrix res(nrow, ncol);
      for (int64_t j = 0; j < ncol; ++j) {
        for (int64_t i = 0; i < nrow; ++i) {
          res(i, j) = out.data[out.offset + i * out.strides[0] + j * out.strides[1]];
        }
      }
      mlirFree(out.basePtr);
      return res;
    }
    Rcpp::stop("unsupported matrix return type");
  }

  Rcpp::stop("unsupported return type");
  return R_NilValue;
}

extern "C" void R_init_rdslmlir_runtime(DllInfo *info) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
  static const R_CallMethodDef callMethods[] = {
      {"rdslmlir_compile", (DL_FUNC)&rdslmlir_compile, 2},
      {"rdslmlir_call", (DL_FUNC)&rdslmlir_call, 2},
      {NULL, NULL, 0}};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(info, FALSE);
}

compile_mlir <- function(module, target = c("cpu", "openmp", "gpu", "linalg"), r_ast_to_mlir = Sys.which("r-ast-to-mlir"), r_mlir_opt = Sys.which("r-mlir-opt"), output = NULL) {
  if (!nzchar(r_ast_to_mlir)) stop("r-ast-to-mlir not found on PATH", call. = FALSE)
  target <- match.arg(target)
  json <- render_json(module, pretty = FALSE)
  json_path <- tempfile(fileext = ".json")
  writeLines(json, json_path, useBytes = TRUE)

  mlir_text <- system2(r_ast_to_mlir, json_path, stdout = TRUE, stderr = TRUE)
  status <- attr(mlir_text, "status") %||% 0
  if (!identical(status, 0L) && !identical(status, 0)) {
    stop("r-ast-to-mlir failed:\n", paste(mlir_text, collapse = "\n"), call. = FALSE)
  }

  if (nzchar(r_mlir_opt)) {
    pass_flag <- switch(target,
      cpu = "--r-to-scf",
      openmp = "--r-to-openmp",
      gpu = "--r-to-gpu",
      linalg = "--r-to-linalg"
    )
    mlir_text <- system2(r_mlir_opt, pass_flag, input = paste(mlir_text, collapse = "\n"), stdout = TRUE, stderr = TRUE)
    status <- attr(mlir_text, "status") %||% 0
    if (!identical(status, 0L) && !identical(status, 0)) {
      stop("r-mlir-opt failed:\n", paste(mlir_text, collapse = "\n"), call. = FALSE)
    }
  }

  if (!is.null(output)) writeLines(mlir_text, output, useBytes = TRUE)
  paste(mlir_text, collapse = "\n")
}

execute_mlir <- function(module, target = c("cpu", "openmp", "gpu"), entry = "main",
                          entry_result = NULL,
                          r_ast_to_mlir = Sys.which("r-ast-to-mlir"),
                          r_mlir_opt = Sys.which("r-mlir-opt"),
                          mlir_opt = Sys.which("mlir-opt"),
                          runner = NULL, runner_shared_libs = NULL, pipeline = NULL, verbose = FALSE) {
  target <- match.arg(target)
  if (!nzchar(r_mlir_opt)) stop("r-mlir-opt not found on PATH", call. = FALSE)
  mlir_text <- compile_mlir(module, target = target, r_ast_to_mlir = r_ast_to_mlir, r_mlir_opt = r_mlir_opt)

  if (!nzchar(mlir_opt)) stop("mlir-opt not found on PATH", call. = FALSE)
  if (is.null(runner)) {
    runner <- if (target == "gpu") Sys.which("mlir-cuda-runner") else Sys.which("mlir-cpu-runner")
    if (!nzchar(runner)) runner <- Sys.which("mlir-runner")
  }
  if (!nzchar(runner)) {
    stop("MLIR runner not found on PATH (mlir-cpu-runner/mlir-cuda-runner/mlir-runner)",
         call. = FALSE)
  }

  if (is.null(runner_shared_libs)) {
    env_libs <- Sys.getenv("MLIR_RUNNER_SHARED_LIBS", "")
    if (nzchar(env_libs)) runner_shared_libs <- env_libs
  }

  if (is.null(pipeline)) {
    pipeline <- switch(target,
      cpu = c(
        "--convert-bufferization-to-memref",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--convert-math-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts"
      ),
      openmp = c(
        "--convert-bufferization-to-memref",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--convert-math-to-llvm",
        "--convert-openmp-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts"
      ),
      gpu = c(
        "--gpu-kernel-outlining",
        "--convert-gpu-to-nvvm",
        "--convert-bufferization-to-memref",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-index-to-llvm",
        "--convert-math-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts"
      )
    )
  }

  lowered <- system2(mlir_opt, pipeline, input = mlir_text, stdout = TRUE, stderr = TRUE)
  status <- attr(lowered, "status") %||% 0
  if (!identical(status, 0L) && !identical(status, 0)) {
    stop("mlir-opt failed:\n", paste(lowered, collapse = "\n"), call. = FALSE)
  }

  runner_args <- c("-e", entry)
  if (!is.null(entry_result)) {
    runner_args <- c(runner_args, paste0("--entry-point-result=", entry_result))
  }
  if (!is.null(runner_shared_libs)) {
    libs <- runner_shared_libs
    if (length(libs) > 1) libs <- paste(libs, collapse = ",")
    runner_args <- c(runner_args, paste0("--shared-libs=", libs))
  }

  out <- system2(runner, runner_args, input = paste(lowered, collapse = "\n"), stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status") %||% 0
  if (!identical(status, 0L) && !identical(status, 0)) {
    stop("runner failed:\n", paste(out, collapse = "\n"), call. = FALSE)
  }
  if (isTRUE(verbose) && length(out) > 0) message(paste(out, collapse = "\n"))
  paste(out, collapse = "\n")
}

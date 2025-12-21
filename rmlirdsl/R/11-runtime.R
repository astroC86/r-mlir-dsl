runtime_lib_name <- function() {
  paste0("librdslmlir_runtime", .Platform$dynlib.ext)
}

runtime_find_lib <- function(path = NULL) {
  if (!is.null(path) && nzchar(path)) return(path)
  env_path <- Sys.getenv("RMLIR_RUNTIME_LIB", "")
  if (nzchar(env_path)) return(env_path)
  pkg_path <- system.file("libs", runtime_lib_name(), package = "rmlirdsl")
  if (nzchar(pkg_path) && file.exists(pkg_path)) return(pkg_path)
  local_path <- file.path(getwd(), "build", "runtime", runtime_lib_name())
  if (file.exists(local_path)) return(local_path)
  stop("rdslmlir runtime shared library not found; set RMLIR_RUNTIME_LIB or pass a path", call. = FALSE)
}

runtime_load <- function(path = NULL) {
  if (!is.null(.rmlirdsl_state$runtime_dll)) {
    return(invisible(path %||% ""))
  }
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp package is required for the rdslmlir runtime", call. = FALSE)
  }
  lib_path <- runtime_find_lib(path)
  dll <- dyn.load(lib_path)
  .rmlirdsl_state$runtime_dll <- dll
  invisible(lib_path)
}

runtime_compile <- function(module, entry = NULL, target = c("cpu")) {
  target <- match.arg(target)
  if (target != "cpu") stop("runtime only supports cpu target for now", call. = FALSE)
  if (inherits(module, "rmlir_func")) module <- rmlir$module(module)
  if (!inherits(module, "rmlir_module")) stop("runtime_compile() requires an rmlir module", call. = FALSE)

  funcs <- module$functions
  if (length(funcs) == 0) stop("runtime_compile() requires at least one function", call. = FALSE)
  if (is.null(entry)) entry <- (attr(funcs[[1]], "rmlir_name") %||% funcs[[1]]$name)

  func <- NULL
  for (f in funcs) {
    fname <- attr(f, "rmlir_name") %||% f$name
    if (identical(fname, entry)) func <- f
  }
  if (is.null(func)) stop("entry function not found: ", entry, call. = FALSE)

  if (is.null(.rmlirdsl_state$runtime_dll)) runtime_load()
  json <- render_json(module, pretty = FALSE)
  compile_sym <- getNativeSymbolInfo("rdslmlir_compile", PACKAGE = .rmlirdsl_state$runtime_dll)
  handle <- .Call(compile_sym, json, entry)

  params <- rmlir_func_field(func, "rmlir_params", "params")
  ret <- rmlir_func_field(func, "rmlir_ret", "ret")
  param_names <- names(params)

  call_fn <- function(...) {
    args <- list(...)
    if (length(args) != length(params)) {
      stop("expected ", length(params), " argument(s), got ", length(args), call. = FALSE)
    }
    if (length(args) > 0 && any(nzchar(names(args)))) {
      if (is.null(param_names) || any(!nzchar(param_names))) {
        stop("named arguments require all params to be named", call. = FALSE)
      }
      ordered <- vector("list", length(params))
      for (i in seq_along(params)) {
        nm <- param_names[[i]]
        if (!nm %in% names(args)) stop("missing argument: ", nm, call. = FALSE)
        ordered[[i]] <- args[[nm]]
      }
      args <- ordered
    }
    call_sym <- getNativeSymbolInfo("rdslmlir_call", PACKAGE = .rmlirdsl_state$runtime_dll)
    .Call(call_sym, handle, args)
  }

  attr(call_fn, "rmlir_params") <- params
  attr(call_fn, "rmlir_ret") <- ret
  attr(call_fn, "rmlir_entry") <- entry
  class(call_fn) <- "rmlir_runtime_fn"
  call_fn
}

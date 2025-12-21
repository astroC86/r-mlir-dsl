rmlir_func <- function(name, params, body, ret = NULL) {
  stopifnot(inherits(params, "rmlir_params"))
  if (!is.function(body)) stop("body must be a function", call. = FALSE)

  wrapper <- function(...) {
    rmlir_require_ctx(paste0(name, "()"))
    args <- list(...)
    args <- normalize_call_args(args, params, name)
    args_exprs <- Map(coerce_expr, args, params)
    if (is.null(ret)) {
      stop("rmlir function '", name, "' has no return type", call. = FALSE)
    }
    ast <- list(type = "Call", name = name, args = lapply(args_exprs, expr_ast))
    rmlir_expr(ast, ret)
  }

  attr(wrapper, "rmlir_name") <- name
  attr(wrapper, "rmlir_params") <- params
  attr(wrapper, "rmlir_body") <- body
  attr(wrapper, "rmlir_ret") <- ret
  class(wrapper) <- c("rmlir_func", "function")
  wrapper
}

Params <- function(...) {
  types <- list(...)
  if (length(types) == 0) stop("Params(...) requires at least one parameter", call. = FALSE)
  if (is.null(names(types)) || any(names(types) == "")) stop("Params(...) requires named arguments", call. = FALSE)
  for (nm in names(types)) if (!inherits(types[[nm]], "rmlir_type")) stop("Param '", nm, "' must be an rmlir type", call. = FALSE)
  structure(types, class = "rmlir_params")
}

fn <- function(name, params, body = NULL, ret = NULL) {
  if (is.null(body)) return(function(body) fn(name, params, body = body, ret = ret))
  rmlir_func(name, params, body, ret = ret)
}

normalize_call_args <- function(args, params, name) {
  if (length(args) != length(params)) {
    stop(name, "() expected ", length(params), " argument(s), got ", length(args), call. = FALSE)
  }
  if (length(args) == 0) return(args)
  param_names <- names(params)
  if (any(nzchar(names(args)))) {
    if (is.null(param_names) || any(!nzchar(param_names))) {
      stop(name, "() named arguments require all params to be named", call. = FALSE)
    }
    ordered <- vector("list", length(params))
    for (i in seq_along(params)) {
      nm <- param_names[[i]]
      if (!nm %in% names(args)) stop(name, "() missing argument: ", nm, call. = FALSE)
      ordered[[i]] <- args[[nm]]
    }
    return(ordered)
  }
  args
}

rmlir_func_field <- function(func, attr_name, list_name) {
  value <- attr(func, attr_name)
  if (!is.null(value)) return(value)
  if (is.list(func)) return(func[[list_name]])
  NULL
}

find_forbidden_calls <- function(expr, forbidden) {
  found <- character()
  walk <- function(node) {
    if (is.call(node)) {
      fun <- node[[1]]
      if (is.symbol(fun)) {
        name <- as.character(fun)
        if (name %in% forbidden) found <<- unique(c(found, name))
      } else if (is.call(fun)) {
        op <- as.character(fun[[1]])
        if (op %in% c("::", ":::")) {
          name <- as.character(fun[[3]])
          if (name %in% forbidden) {
            pkg <- as.character(fun[[2]])
            found <<- unique(c(found, paste0(pkg, "::", name)))
          }
        }
      }
      for (i in seq_along(node)) walk(node[[i]])
    } else if (is.pairlist(node)) {
      for (el in node) walk(el)
    }
  }
  walk(expr)
  found
}

find_for_indices <- function(expr) {
  indices <- character()
  walk <- function(node) {
    if (is.call(node)) {
      fun <- node[[1]]
      is_for <- FALSE
      if (is.symbol(fun) && identical(as.character(fun), "for_")) {
        is_for <- TRUE
      } else if (is.call(fun) && identical(as.character(fun[[1]]), "$")) {
        if (length(fun) >= 3 && is.symbol(fun[[2]]) && is.symbol(fun[[3]])) {
          if (identical(as.character(fun[[2]]), "rmlir") &&
              identical(as.character(fun[[3]]), "for_")) {
            is_for <- TRUE
          }
        }
      }
      if (is_for && length(node) >= 2 && is.symbol(node[[2]])) {
        indices <<- unique(c(indices, as.character(node[[2]])))
      }
      for (i in seq_along(node)) walk(node[[i]])
    } else if (is.pairlist(node)) {
      for (el in node) walk(el)
    }
  }
  walk(expr)
  indices
}

find_illegal_calls <- function(expr, allowed_symbols, allowed_functions, allowed_rmlir_calls, env) {
  illegal <- character()
  is_rmlir_func_name <- function(name) {
    if (!exists(name, envir = env, inherits = TRUE)) return(FALSE)
    inherits(get(name, envir = env, inherits = TRUE), "rmlir_func")
  }
  walk <- function(node) {
    if (is.call(node)) {
      fun <- node[[1]]
      if (is.symbol(fun)) {
        name <- as.character(fun)
        if (name == "$") {
          # Allow field access like ty$index in arguments.
        } else if (!(name %in% allowed_symbols) && !(name %in% allowed_functions) &&
                   !is_rmlir_func_name(name)) {
          illegal <<- unique(c(illegal, name))
        }
      } else if (is.call(fun)) {
        op <- as.character(fun[[1]])
        if (op %in% c("::", ":::")) {
          pkg <- as.character(fun[[2]])
          name <- as.character(fun[[3]])
          illegal <<- unique(c(illegal, paste0(pkg, "::", name)))
        } else if (op == "$") {
          base <- fun[[2]]
          field <- fun[[3]]
          if (is.symbol(base) && is.symbol(field)) {
            base_name <- as.character(base)
            field_name <- as.character(field)
            if (base_name == "rmlir") {
              if (!(field_name %in% allowed_rmlir_calls)) {
                illegal <<- unique(c(illegal, paste0("rmlir$", field_name)))
              }
            }
          }
        } else {
          illegal <<- unique(c(illegal, op))
        }
      }
      for (i in seq_along(node)) {
        if (i == 1) next
        walk(node[[i]])
      }
    } else if (is.pairlist(node)) {
      for (el in node) walk(el)
    }
  }
  walk(expr)
  illegal
}

validate_func_body <- function(body_fn, params) {
  if (!is.function(body_fn)) return(invisible(NULL))
  if (!requireNamespace("codetools", quietly = TRUE)) return(invisible(NULL))

  allowed_symbols <- c("{", "(", "<-", "<<-", "=", "+", "-", "*", "/", "^", "%%", "%/%", "%*%",
                       ":", "[", "[[", "if", "for", "while", "repeat", "break", "next",
                       "&&", "||", "&", "|", "!", "==", "!=", "<", "<=", ">", ">=")
  allowed_functions <- c("for_", "ret", "const", "var", "let", "index", "slice", "matmul",
                         "alloc_tensor", "clone", "to_tensor", "to_buffer",
                         "materialize_in_destination", "dealloc", "dealloc_tensor",
                         "rmlir_range", "dim", "store",
                         "abs", "sqrt", "exp", "log", "log10", "log2", "sign",
                         "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                         "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                         "round", "floor", "ceiling", "trunc", "signif")
  allowed_rmlir_calls <- c("for_", "ret", "const", "var", "let", "index", "slice", "matmul",
                           "alloc_tensor", "clone", "to_tensor", "to_buffer",
                           "materialize_in_destination", "dealloc", "dealloc_tensor",
                           "range", "dim", "store")

  illegal_calls <- find_illegal_calls(body(body_fn), allowed_symbols, allowed_functions,
                                      allowed_rmlir_calls, environment(body_fn))
  if (length(illegal_calls) > 0) {
    stop("unsupported call(s) in rmlir$fn: ", paste(illegal_calls, collapse = ", "),
         call. = FALSE)
  }

  globals <- codetools::findGlobals(body_fn, merge = FALSE)
  vars <- globals$variables %||% character()
  vars <- setdiff(vars, c("...", "rmlir", "ty", names(params),
                          "NULL", "TRUE", "FALSE", "NA", "NA_integer_",
                          "NA_real_", "NA_character_", "NA_complex_"))
  vars <- setdiff(vars, find_for_indices(body(body_fn)))
  if (length(vars) > 0) {
    env <- environment(body_fn)
    is_rmlir_func <- vapply(vars, function(name) {
      if (!exists(name, envir = env, inherits = TRUE)) return(FALSE)
      inherits(get(name, envir = env, inherits = TRUE), "rmlir_func")
    }, logical(1))
    vars <- vars[!is_rmlir_func]
  }
  if (length(vars) > 0) {
    stop("unsupported global reference(s): ", paste(vars, collapse = ", "), call. = FALSE)
  }
  invisible(NULL)
}

compile_func_body <- function(func) {
  func_name <- rmlir_func_field(func, "rmlir_name", "name")
  params <- rmlir_func_field(func, "rmlir_params", "params")
  func_body <- rmlir_func_field(func, "rmlir_body", "body")
  ret <- rmlir_func_field(func, "rmlir_ret", "ret")
  if (is.null(func_name) || is.null(params) || is.null(func_body)) {
    stop("compile_func_body() requires an rmlir function", call. = FALSE)
  }

  validate_func_body(func_body, params)

  ctx <- rmlir_ctx()
  old <- .rmlirdsl_state$ctx
  .rmlirdsl_state$ctx <- ctx
  on.exit({ .rmlirdsl_state$ctx <- old }, add = TRUE)

  args <- list()
  for (nm in names(params)) {
    pty <- params[[nm]]
    expr <- rmlir_expr(list(type = "Var", name = nm), pty)
    args[[nm]] <- expr
    ctx$bind(nm, expr)
  }

  do.call(func_body, args)

  params_ast <- lapply(names(params), function(nm) list(name = nm, type = rmlir_type_to_ast(params[[nm]])))
  list(
    type = "Function",
    name = func_name,
    params = params_ast,
    returnType = if (is.null(ret)) NULL else rmlir_type_to_ast(ret),
    body = ctx$stmts
  )
}

module <- function(...) {
  funcs <- list(...)
  if (length(funcs) == 1 && is.list(funcs[[1]]) && !inherits(funcs[[1]], "rmlir_func")) {
    funcs <- funcs[[1]]
  }
  for (f in funcs) if (!inherits(f, "rmlir_func")) stop("module() expects rmlir functions", call. = FALSE)
  structure(list(functions = funcs), class = "rmlir_module")
}

render_json <- function(module, pretty = TRUE) {
  if (!inherits(module, "rmlir_module")) stop("render_json() requires an rmlir module", call. = FALSE)
  funcs <- lapply(module$functions, compile_func_body)
  mod_ast <- list(type = "Module", functions = funcs)
  jsonlite::toJSON(mod_ast, auto_unbox = TRUE, pretty = isTRUE(pretty), na = "null")
}

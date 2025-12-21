for_ <- function(sym, start, end, step = NULL, parallel = FALSE, target = NULL, body) {
  expr_quoted <- substitute(body)
  ctx <- rmlir_require_ctx("for_()")
  sym <- substitute(sym)
  if (!is.symbol(sym)) stop("for_() first argument must be a symbol", call. = FALSE)
  name <- as.character(sym)

  if (is.null(step)) step <- const(1, ty$index)
  start <- coerce_expr(start, ty$index)
  end <- coerce_expr(end, ty$index)
  step <- coerce_expr(step, ty$index)

  inner <- rmlir_ctx(parent = ctx$symbols)
  old <- .rmlirdsl_state$ctx
  .rmlirdsl_state$ctx <- inner
  on.exit({ .rmlirdsl_state$ctx <- old }, add = TRUE)

  idx_expr <- rmlir_expr(list(type = "Var", name = name), ty$index)
  inner$bind(name, idx_expr)
  env <- new.env(parent = parent.frame())
  env[[name]] <- idx_expr
  eval(expr_quoted, envir = env)

  stmt <- list(
    type = "For",
    index = name,
    start = expr_ast(start),
    end = expr_ast(end),
    step = expr_ast(step),
    parallel = isTRUE(parallel),
    body = inner$stmts
  )
  if (!is.null(target)) stmt$target <- as.character(target)
  ctx$emit(stmt)
  invisible(NULL)
}

ret <- function(value = NULL) {
  ctx <- rmlir_require_ctx("ret()")
  if (is.null(value)) {
    ctx$emit(list(type = "Return"))
  } else {
    ctx$emit(list(type = "Return", value = expr_ast(value)))
  }
  invisible(NULL)
}

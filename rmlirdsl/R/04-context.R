rmlir_ctx <- function(parent = NULL) {
  ctx <- new.env(parent = emptyenv())
  ctx$stmts <- list()
  ctx$symbols <- new.env(parent = parent %||% emptyenv())
  ctx$emit <- function(stmt) {
    ctx$stmts[[length(ctx$stmts) + 1L]] <- stmt
    invisible(NULL)
  }
  ctx$bind <- function(name, expr) {
    assign(name, expr, envir = ctx$symbols)
    invisible(NULL)
  }
  ctx
}

rmlir_require_ctx <- function(feature) {
  ctx <- .rmlirdsl_state$ctx
  if (is.null(ctx)) stop(feature, " used outside rmlirdsl render context", call. = FALSE)
  ctx
}

let <- function(sym, value, type = NULL) {
  sym <- substitute(sym)
  if (!is.symbol(sym)) stop("let() first argument must be a symbol", call. = FALSE)
  name <- as.character(sym)
  ctx <- rmlir_require_ctx("let()")
  if (is.null(type)) {
    if (is.null(value$type)) stop("let() requires a type or a typed value", call. = FALSE)
    type <- value$type
  }
  stmt <- list(type = "Assign", name = name, value = expr_ast(value))
  ctx$emit(stmt)
  out <- rmlir_expr(list(type = "Var", name = name), type)
  ctx$bind(name, out)
  out
}

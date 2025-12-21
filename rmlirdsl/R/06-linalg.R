`%*%.rmlir_expr` <- function(x, y) {
  matmul(x, y)
}

matmul <- function(lhs, rhs) {
  lhs_expr <- coerce_expr(lhs)
  rhs_expr <- coerce_expr(rhs)

  if (is.null(lhs_expr$type) || lhs_expr$type$kind != "tensor") {
    stop("matmul() lhs must be a tensor expression", call. = FALSE)
  }
  if (is.null(rhs_expr$type) || rhs_expr$type$kind != "tensor") {
    stop("matmul() rhs must be a tensor expression", call. = FALSE)
  }
  require_memref(lhs_expr, "matmul()")
  require_memref(rhs_expr, "matmul()")
  if (!identical(lhs_expr$type$elem$name, rhs_expr$type$elem$name)) {
    stop("matmul() element types must match", call. = FALSE)
  }

  lhs_shape <- lhs_expr$type$shape %||% rep(NA_integer_, 2)
  rhs_shape <- rhs_expr$type$shape %||% rep(NA_integer_, 2)
  if (length(lhs_shape) != 2 || length(rhs_shape) != 2) {
    stop("matmul() expects rank-2 tensors (matrices)", call. = FALSE)
  }

  out_shape <- c(lhs_shape[[1]], rhs_shape[[2]])
  out_type <- ty$tensor(lhs_expr$type$elem, shape = out_shape)
  ast <- list(type = "MatMul", lhs = expr_ast(lhs_expr), rhs = expr_ast(rhs_expr))
  rmlir_expr(ast, out_type)
}

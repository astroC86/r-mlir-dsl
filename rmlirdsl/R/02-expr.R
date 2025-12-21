rmlir_expr <- function(ast, type = NULL) {
  structure(list(ast = ast, type = type), class = "rmlir_expr")
}

expr_ast <- function(x) {
  if (inherits(x, "rmlir_expr")) return(x$ast)
  if (is.numeric(x) && length(x) == 1) return(list(type = "Number", value = x))
  if (is.logical(x) && length(x) == 1) return(list(type = "Bool", value = isTRUE(x)))
  stop("Unsupported expression value", call. = FALSE)
}

coerce_expr <- function(x, type_hint = NULL) {
  if (inherits(x, "rmlir_expr")) return(x)
  if (!is.null(type_hint) && inherits(type_hint, "rmlir_type") && type_hint$kind == "tensor") {
    type_hint <- type_hint$elem
  }
  if (is.numeric(x) && length(x) == 1) {
    ty_hint <- type_hint %||% (if (is.integer(x)) ty$i32 else ty$f64)
    return(rmlir_expr(list(type = "Number", value = x, dtype = rmlir_type_to_ast(ty_hint)), ty_hint))
  }
  if (is.logical(x) && length(x) == 1) {
    return(rmlir_expr(list(type = "Bool", value = isTRUE(x), dtype = rmlir_type_to_ast(ty$bool)), ty$bool))
  }
  stop("Unsupported expression value", call. = FALSE)
}

const <- function(value, type = NULL) {
  if (is.null(type)) {
    return(coerce_expr(value))
  }
  if (!inherits(type, "rmlir_type") || type$kind != "prim") {
    stop("const() requires a primitive type", call. = FALSE)
  }
  if (type$name == "bool") {
    ast <- list(type = "Bool", value = isTRUE(value), dtype = rmlir_type_to_ast(type))
  } else {
    ast <- list(type = "Number", value = value, dtype = rmlir_type_to_ast(type))
  }
  rmlir_expr(ast, type)
}

var <- function(name) {
  if (!is.character(name) || length(name) != 1 || !nzchar(name)) {
    stop("var() requires a non-empty string", call. = FALSE)
  }
  ctx <- .rmlirdsl_state$ctx
  type <- NULL
  if (!is.null(ctx) && exists(name, envir = ctx$symbols, inherits = TRUE)) {
    type <- get(name, envir = ctx$symbols)$type
  }
  rmlir_expr(list(type = "Var", name = name), type)
}

Ops.rmlir_expr <- function(e1, e2 = NULL) {
  op <- .Generic

  if (missing(e2)) {
    if (op %in% c("+", "-")) {
      ast <- list(type = "Unary", op = op, value = expr_ast(e1))
      return(rmlir_expr(ast, e1$type))
    }
    stop("Unsupported unary op for rmlir_expr: ", op, call. = FALSE)
  }

  if (!(op %in% c("+", "-", "*", "/"))) {
    stop("Unsupported op for rmlir_expr: ", op, call. = FALSE)
  }

  left <- coerce_expr(e1)
  right <- coerce_expr(e2, left$type)
  out_type <- left$type %||% right$type
  ast <- list(type = "Binary", op = op, lhs = expr_ast(left), rhs = expr_ast(right))
  rmlir_expr(ast, out_type)
}

math_elem_type <- function(expr) {
  if (is.null(expr$type) || !inherits(expr$type, "rmlir_type")) return(NULL)
  if (expr$type$kind == "prim") return(expr$type)
  if (expr$type$kind == "tensor") return(expr$type$elem)
  NULL
}

math_arg_scalar <- function(args, label, op, default = NULL) {
  if (length(args) == 0) return(default)
  if (length(args) > 1) stop(op, "() expects at most one ", label, " argument", call. = FALSE)
  if (!is.null(names(args)) && nzchar(names(args)[1]) && names(args)[1] != label) {
    stop(op, "() expects argument '", label, "'", call. = FALSE)
  }
  args[[1]]
}

math_require_float <- function(expr, op) {
  elem <- math_elem_type(expr)
  if (is.null(elem) || elem$kind != "prim" || elem$name != "f64") {
    stop(op, "() expects a f64 expression", call. = FALSE)
  }
  invisible(elem)
}

math_require_numeric <- function(expr, op) {
  elem <- math_elem_type(expr)
  if (is.null(elem) || elem$kind != "prim" || !(elem$name %in% c("i32", "f64"))) {
    stop(op, "() expects a numeric expression", call. = FALSE)
  }
  invisible(elem)
}

Math.rmlir_expr <- function(x, ...) {
  op <- .Generic
  args <- list(...)
  x_expr <- coerce_expr(x)

  float_ops <- c("sqrt", "exp", "log", "log10", "log2",
                 "sin", "cos", "tan", "asin", "acos", "atan",
                 "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                 "round", "floor", "ceiling", "trunc", "signif")
  if (op %in% float_ops) {
    math_require_float(x_expr, op)
  } else if (op %in% c("abs", "sign")) {
    math_require_numeric(x_expr, op)
  } else {
    stop("Unsupported Math op for rmlir_expr: ", op, call. = FALSE)
  }

  base <- NULL
  digits <- NULL
  if (op == "log") {
    base <- math_arg_scalar(args, "base", op)
    if (!is.null(base) && (!is.numeric(base) || length(base) != 1 || !is.finite(base))) {
      stop("log() base must be a numeric scalar", call. = FALSE)
    }
  } else if (op %in% c("round", "signif")) {
    default_digits <- if (op == "signif") 6 else 0
    digits <- math_arg_scalar(args, "digits", op, default = default_digits)
    if (!is.numeric(digits) || length(digits) != 1 || !is.finite(digits)) {
      stop(op, "() digits must be a numeric scalar", call. = FALSE)
    }
    if (abs(digits - round(digits)) > 1e-8) {
      stop(op, "() digits must be an integer", call. = FALSE)
    }
    digits <- as.integer(digits)
  } else if (length(args) > 0) {
    stop(op, "() does not accept extra arguments", call. = FALSE)
  }

  ast_args <- list(expr_ast(x_expr))
  if (!is.null(base)) ast_args <- c(ast_args, expr_ast(as.numeric(base)))
  if (!is.null(digits)) ast_args <- c(ast_args, expr_ast(as.numeric(digits)))
  ast <- list(type = "Call", name = op, args = ast_args)
  rmlir_expr(ast, x_expr$type)
}

Math2.rmlir_expr <- function(x, y) {
  op <- .Generic
  if (op != "atan2") stop("Unsupported Math2 op for rmlir_expr: ", op, call. = FALSE)
  left <- coerce_expr(x)
  right <- coerce_expr(y, left$type)
  math_require_float(left, op)
  ast <- list(type = "Call", name = op, args = list(expr_ast(left), expr_ast(right)))
  rmlir_expr(ast, left$type %||% right$type)
}

atan2 <- function(y, x) {
  if (inherits(y, "rmlir_expr") || inherits(x, "rmlir_expr")) {
    left <- coerce_expr(y)
    right <- coerce_expr(x, left$type)
    math_require_float(left, "atan2")
    math_require_float(right, "atan2")
    ast <- list(type = "Call", name = "atan2", args = list(expr_ast(left), expr_ast(right)))
    return(rmlir_expr(ast, left$type %||% right$type))
  }
  base::atan2(y, x)
}

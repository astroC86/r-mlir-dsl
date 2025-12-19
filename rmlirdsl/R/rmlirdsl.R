.rmlirdsl_state <- new.env(parent = emptyenv())
.rmlirdsl_state$ctx <- NULL
.rmlirdsl_state$runtime_dll <- NULL

`%||%` <- function(x, y) if (is.null(x)) y else x

rmlir_type <- function(kind, ...) {
  structure(list(kind = kind, ...), class = "rmlir_type")
}

ty <- new.env(parent = emptyenv())
ty$i32 <- rmlir_type("prim", name = "i32")
ty$f64 <- rmlir_type("prim", name = "f64")
ty$bool <- rmlir_type("prim", name = "bool")
ty$index <- rmlir_type("prim", name = "index")

ty$tensor <- function(elem, shape = NULL) {
  if (!inherits(elem, "rmlir_type") || elem$kind != "prim") {
    stop("tensor() elem must be a primitive rmlir_type", call. = FALSE)
  }
  if (!is.null(shape)) {
    if (!is.numeric(shape)) stop("tensor() shape must be numeric or NULL", call. = FALSE)
    shape <- as.integer(shape)
  }
  rmlir_type("tensor", elem = elem, shape = shape, storage = "memref")
}

ty$tensor_value <- function(elem, shape = NULL) {
  if (!inherits(elem, "rmlir_type") || elem$kind != "prim") {
    stop("tensor_value() elem must be a primitive rmlir_type", call. = FALSE)
  }
  if (!is.null(shape)) {
    if (!is.numeric(shape)) stop("tensor_value() shape must be numeric or NULL", call. = FALSE)
    shape <- as.integer(shape)
  }
  rmlir_type("tensor", elem = elem, shape = shape, storage = "tensor")
}

ty$vector <- function(elem, n = NULL) {
  if (is.null(n)) return(ty$tensor(elem, shape = NA_integer_))
  ty$tensor(elem, shape = as.integer(n))
}

ty$vector_value <- function(elem, n = NULL) {
  if (is.null(n)) return(ty$tensor_value(elem, shape = NA_integer_))
  ty$tensor_value(elem, shape = as.integer(n))
}

ty$matrix <- function(elem, nrow = NULL, ncol = NULL) {
  shape <- c(if (is.null(nrow)) NA_integer_ else as.integer(nrow),
             if (is.null(ncol)) NA_integer_ else as.integer(ncol))
  ty$tensor(elem, shape = shape)
}

ty$matrix_value <- function(elem, nrow = NULL, ncol = NULL) {
  shape <- c(if (is.null(nrow)) NA_integer_ else as.integer(nrow),
             if (is.null(ncol)) NA_integer_ else as.integer(ncol))
  ty$tensor_value(elem, shape = shape)
}

rmlir_type_to_ast <- function(type) {
  stopifnot(inherits(type, "rmlir_type"))
  if (type$kind == "prim") return(type$name)
  if (type$kind == "tensor") {
    shape <- type$shape
    if (is.null(shape)) shape <- NA_integer_
    storage <- type$storage %||% "memref"
    return(list(type = "tensor", elem = type$elem$name, shape = shape, storage = storage))
  }
  stop("Unsupported type kind: ", type$kind, call. = FALSE)
}

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

tensor_storage <- function(type) {
  if (!inherits(type, "rmlir_type") || type$kind != "tensor") return(NULL)
  type$storage %||% "memref"
}

require_memref <- function(expr, feature) {
  storage <- tensor_storage(expr$type)
  if (!is.null(storage) && storage != "memref") {
    stop(feature, " requires a buffer; use rmlir$to_buffer() first", call. = FALSE)
  }
}

require_tensor_value <- function(expr, feature) {
  storage <- tensor_storage(expr$type)
  if (!is.null(storage) && storage != "tensor") {
    stop(feature, " requires a tensor value; use rmlir$to_tensor() first", call. = FALSE)
  }
}

tensor_type_with_storage <- function(type, storage) {
  if (!inherits(type, "rmlir_type") || type$kind != "tensor") {
    stop("expected tensor type", call. = FALSE)
  }
  rmlir_type("tensor", elem = type$elem, shape = type$shape, storage = storage)
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

rmlir_range <- function(start, end, step = NULL, size_hint = NA_integer_) {
  if (is.null(step)) step <- const(1, ty$index)
  start_expr <- coerce_expr(start, ty$index)
  end_expr <- coerce_expr(end, ty$index)
  step_expr <- coerce_expr(step, ty$index)

  size_expr <- (end_expr - start_expr) / step_expr + const(1, ty$index)

  if (is.na(size_hint)) {
    s <- if (is.numeric(start) && length(start) == 1 && is.finite(start)) as.numeric(start) else NULL
    e <- if (is.numeric(end) && length(end) == 1 && is.finite(end)) as.numeric(end) else NULL
    st <- if (is.numeric(step) && length(step) == 1 && is.finite(step)) as.numeric(step) else NULL
    if (!is.null(s) && !is.null(e)) {
      if (is.null(st)) st <- 1
      if (st != 0) size_hint <- as.integer(length(seq.int(from = s, to = e, by = st)))
    }
  }

  structure(
    list(start = start_expr, end = end_expr, step = step_expr, size = size_expr, size_hint = size_hint),
    class = "rmlir_range"
  )
}

`:.rmlir_expr` <- function(e1, e2) {
  rmlir_range(e1, e2)
}

dim_expr <- function(target, axis) {
  target_expr <- coerce_expr(target)
  if (is.null(target_expr$type) || target_expr$type$kind != "tensor") {
    stop("dim() target must be a tensor expression", call. = FALSE)
  }
  require_memref(target_expr, "dim()")
  if (!is.numeric(axis) || length(axis) != 1 || !is.finite(axis)) {
    stop("dim() axis must be a finite numeric scalar", call. = FALSE)
  }
  if (axis < 1) stop("dim() axis is 1-based", call. = FALSE)
  axis0 <- as.integer(axis) - 1L
  ast <- list(type = "Dim", target = expr_ast(target_expr), axis = axis0)
  rmlir_expr(ast, ty$index)
}

rmlir_dims <- function(dims) {
  structure(dims, class = "rmlir_dims")
}

dim.rmlir_expr <- function(x) {
  if (is.null(x$type) || x$type$kind != "tensor") {
    stop("dim() target must be a tensor expression", call. = FALSE)
  }
  shape <- x$type$shape
  rank <- if (!is.null(shape)) length(shape) else 1L
  dims <- vector("list", rank)
  for (i in seq_len(rank)) {
    known <- !is.null(shape) && length(shape) >= i && !is.na(shape[[i]])
    if (known) {
      dims[[i]] <- const(as.integer(shape[[i]]), ty$index)
    } else {
      dims[[i]] <- dim_expr(x, i)
    }
  }
  rmlir_dims(dims)
}

`[.rmlir_dims` <- function(x, i, drop = TRUE) {
  out <- unclass(x)[i]
  if (isTRUE(drop) && length(out) == 1) return(out[[1]])
  rmlir_dims(out)
}

`[[.rmlir_dims` <- function(x, i, ...) {
  unclass(x)[[i]]
}

length.rmlir_dims <- function(x) {
  length(unclass(x))
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

`%*%.rmlir_expr` <- function(x, y) {
  matmul(x, y)
}

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

size_hint_from_expr <- function(expr) {
  if (inherits(expr, "rmlir_expr") && is.list(expr$ast) && identical(expr$ast$type, "Number")) {
    val <- expr$ast$value
    if (is.numeric(val) && length(val) == 1 && is.finite(val)) return(as.integer(val))
  }
  if (is.numeric(expr) && length(expr) == 1 && is.finite(expr)) return(as.integer(expr))
  NA_integer_
}

range_from_numeric_vector <- function(vec) {
  if (!is.numeric(vec) || length(vec) < 2) return(NULL)
  step <- vec[2] - vec[1]
  if (!is.finite(step) || step == 0) return(NULL)
  if (length(vec) > 2) {
    if (any(diff(vec) != step)) return(NULL)
  }
  rmlir_range(vec[1], vec[length(vec)], step, size_hint = as.integer(length(vec)))
}

build_slice_expr <- function(target_expr, offsets_exprs, sizes_exprs, strides_exprs, size_hints = NULL, drop_dims = NULL) {
  offsets_ast <- lapply(offsets_exprs, expr_ast)
  sizes_ast <- lapply(sizes_exprs, expr_ast)
  strides_ast <- lapply(strides_exprs, expr_ast)

  if (is.null(size_hints)) {
    size_hints <- vapply(sizes_exprs, size_hint_from_expr, NA_integer_)
  }
  size_hints <- as.integer(size_hints)

  if (is.null(drop_dims)) drop_dims <- rep(FALSE, length(size_hints))
  if (length(drop_dims) != length(size_hints)) {
    stop("slice() drop_dims length mismatch", call. = FALSE)
  }

  out_shape <- size_hints[!drop_dims]
  out_type <- ty$tensor(target_expr$type$elem, shape = out_shape)
  ast <- list(
    type = "Slice",
    source = expr_ast(target_expr),
    offsets = offsets_ast,
    sizes = sizes_ast,
    strides = strides_ast,
    dropDims = drop_dims
  )
  rmlir_expr(ast, out_type)
}

parse_index_arg <- function(arg) {
  if (inherits(arg, "rmlir_range")) {
    step_val <- size_hint_from_expr(arg$step)
    start_val <- size_hint_from_expr(arg$start)
    if (!is.na(step_val) && step_val <= 0) {
      stop("index() only supports positive range steps", call. = FALSE)
    }
    if (!is.na(start_val) && start_val < 1) {
      stop("index() indices are 1-based", call. = FALSE)
    }
    return(list(
      kind = "slice",
      offset = arg$start,
      size = arg$size,
      stride = arg$step,
      size_hint = arg$size_hint
    ))
  }

  if (is.numeric(arg) && length(arg) > 1) {
    if (any(arg <= 0)) stop("index() indices are 1-based", call. = FALSE)
    rng <- range_from_numeric_vector(arg)
    if (is.null(rng)) {
      stop("index() only supports contiguous ranges for slicing; use `:` for dynamic slices", call. = FALSE)
    }
    return(list(
      kind = "slice",
      offset = rng$start,
      size = rng$size,
      stride = rng$step,
      size_hint = rng$size_hint
    ))
  }

  if (is.numeric(arg) && length(arg) == 1) {
    if (arg <= 0) stop("index() indices are 1-based", call. = FALSE)
    return(list(kind = "point", expr = coerce_expr(arg, ty$index)))
  }

  if (inherits(arg, "rmlir_expr")) {
    if (!is.null(arg$type) && (!identical(arg$type$kind, "prim") || !identical(arg$type$name, "index"))) {
      stop("index() requires index-typed expressions", call. = FALSE)
    }
    return(list(kind = "point", expr = arg))
  }

  stop("index() unsupported index argument; use `:` for slicing", call. = FALSE)
}

index <- function(target, ..., drop = TRUE) {
  args <- list(...)
  if (length(args) == 1 && is.list(args[[1]]) &&
      !inherits(args[[1]], "rmlir_expr") && !inherits(args[[1]], "rmlir_range")) {
    args <- args[[1]]
  }
  target_expr <- coerce_expr(target)
  if (is.null(target_expr$type) || target_expr$type$kind != "tensor") {
    stop("index() target must be a tensor expression", call. = FALSE)
  }
  require_memref(target_expr, "index()")

  if (length(args) == 0) stop("index() requires at least one index", call. = FALSE)
  target_rank <- if (!is.null(target_expr$type$shape)) length(target_expr$type$shape) else length(args)
  if (length(args) != target_rank) {
    stop("index() expects ", target_rank, " index argument(s) for this tensor", call. = FALSE)
  }

  parsed <- vector("list", length(args))
  for (i in seq_along(args)) {
    arg <- args[[i]]
    if (is.null(arg)) {
      dim_hint <- NA_integer_
      if (!is.null(target_expr$type$shape) && length(target_expr$type$shape) >= i) {
        dim_hint <- target_expr$type$shape[[i]]
      }
      end_expr <- if (!is.na(dim_hint)) const(dim_hint, ty$index) else dim_expr(target_expr, i)
      rng <- rmlir_range(const(1, ty$index), end_expr, const(1, ty$index), size_hint = dim_hint)
      parsed[[i]] <- list(
        kind = "slice",
        offset = rng$start,
        size = rng$size,
        stride = rng$step,
        size_hint = rng$size_hint,
        is_full = TRUE
      )
    } else {
      parsed[[i]] <- parse_index_arg(arg)
    }
  }
  has_slice <- any(vapply(parsed, function(x) identical(x$kind, "slice"), logical(1)))

  if (has_slice) {
    offsets <- vector("list", length(parsed))
    sizes <- vector("list", length(parsed))
    strides <- vector("list", length(parsed))
    size_hints <- rep(NA_integer_, length(parsed))
    drop_dims <- rep(FALSE, length(parsed))

    for (i in seq_along(parsed)) {
      arg <- parsed[[i]]
      if (identical(arg$kind, "slice")) {
        offsets[[i]] <- arg$offset - const(1, ty$index)
        sizes[[i]] <- arg$size
        strides[[i]] <- arg$stride
        size_hints[i] <- arg$size_hint
      } else {
        offsets[[i]] <- coerce_expr(arg$expr, ty$index) - const(1, ty$index)
        sizes[[i]] <- const(1, ty$index)
        strides[[i]] <- const(1, ty$index)
        size_hints[i] <- 1L
        if (isTRUE(drop)) drop_dims[i] <- TRUE
      }
    }

    return(build_slice_expr(target_expr, offsets, sizes, strides, size_hints, drop_dims))
  }

  indices <- lapply(parsed, function(arg) expr_ast(coerce_expr(arg$expr, ty$index) - const(1, ty$index)))
  ast <- list(type = "Index", target = expr_ast(target_expr), indices = indices)
  rmlir_expr(ast, target_expr$type$elem)
}

`[.rmlir_expr` <- function(x, i, j, ..., drop = TRUE) {
  extra <- as.list(substitute(list(...)))[-1]
  if (length(extra) > 0) stop("only vector/matrix indexing supported", call. = FALSE)

  resolve_index_arg <- function(expr, env) {
    if (is.null(expr)) return(NULL)
    if (is.call(expr) && identical(expr[[1]], as.name(":")) && length(expr) == 3) {
      start <- eval(expr[[2]], env)
      end <- eval(expr[[3]], env)
      return(rmlir_range(start, end))
    }
    eval(expr, env)
  }

  rank <- if (!is.null(x$type$shape)) length(x$type$shape) else if (missing(j)) 1L else 2L
  if (rank == 1) {
    if (!missing(j)) stop("vector indexing only accepts one dimension", call. = FALSE)
    idx <- if (missing(i)) NULL else resolve_index_arg(substitute(i), parent.frame())
    return(index(x, idx, drop = drop))
  }
  if (rank == 2) {
    idx_i <- if (missing(i)) NULL else resolve_index_arg(substitute(i), parent.frame())
    idx_j <- if (missing(j)) NULL else resolve_index_arg(substitute(j), parent.frame())
    return(index(x, idx_i, idx_j, drop = drop))
  }
  stop("indexing only supports rank-1/2 tensors currently", call. = FALSE)
}

`[<-.rmlir_expr` <- function(x, i, j, ..., value) {
  if (missing(value)) stop("assignment requires a value", call. = FALSE)
  extra <- as.list(substitute(list(...)))[-1]
  if (length(extra) > 0) stop("only vector/matrix indexing supported", call. = FALSE)

  resolve_index_arg <- function(expr, env) {
    if (is.null(expr)) return(NULL)
    if (is.call(expr) && identical(expr[[1]], as.name(":")) && length(expr) == 3) {
      start <- eval(expr[[2]], env)
      end <- eval(expr[[3]], env)
      return(rmlir_range(start, end))
    }
    eval(expr, env)
  }

  rank <- if (!is.null(x$type$shape)) length(x$type$shape) else if (missing(j)) 1L else 2L
  if (rank == 1) {
    if (missing(i)) {
      stop("assignment only supports scalar indices for now", call. = FALSE)
    }
    idx <- resolve_index_arg(substitute(i), parent.frame())
    if (is.null(idx)) {
      stop("assignment only supports scalar indices for now", call. = FALSE)
    }
    store(x, idx, value = value)
    return(x)
  }
  if (rank == 2) {
    if (missing(i) || missing(j)) {
      stop("assignment only supports scalar indices for now", call. = FALSE)
    }
    idx_i <- resolve_index_arg(substitute(i), parent.frame())
    idx_j <- resolve_index_arg(substitute(j), parent.frame())
    if (is.null(idx_i) || is.null(idx_j)) {
      stop("assignment only supports scalar indices for now", call. = FALSE)
    }
    store(x, idx_i, idx_j, value = value)
    return(x)
  }
  stop("indexing only supports rank-1/2 tensors currently", call. = FALSE)
}

slice <- function(target, offsets, sizes, strides = NULL) {
  target_expr <- coerce_expr(target)
  if (is.null(target_expr$type) || target_expr$type$kind != "tensor") {
    stop("slice() target must be a tensor expression", call. = FALSE)
  }
  require_memref(target_expr, "slice()")

  normalize_index_list <- function(x) {
    if (inherits(x, "rmlir_range")) {
      stop("slice() requires explicit offsets/sizes; use `:` with index() instead", call. = FALSE)
    }
    if (inherits(x, "rmlir_expr")) return(list(x))
    if (is.list(x)) return(x)
    as.list(x)
  }

  offsets <- normalize_index_list(offsets)
  sizes <- normalize_index_list(sizes)
  if (!is.null(strides)) strides <- normalize_index_list(strides)

  if (length(offsets) != length(sizes)) {
    stop("slice() offsets and sizes must have the same length", call. = FALSE)
  }

  if (is.null(strides)) {
    strides <- rep(list(const(1, ty$index)), length(sizes))
  }
  if (length(strides) != length(sizes)) {
    stop("slice() strides must match offsets/sizes length", call. = FALSE)
  }

  offsets_exprs <- lapply(offsets, function(x) coerce_expr(x, ty$index))
  sizes_exprs <- lapply(sizes, function(x) coerce_expr(x, ty$index))
  strides_exprs <- lapply(strides, function(x) coerce_expr(x, ty$index))
  size_hints <- vapply(sizes, size_hint_from_expr, NA_integer_)
  build_slice_expr(target_expr, offsets_exprs, sizes_exprs, strides_exprs, size_hints)
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

alloc_tensor <- function(shape = NULL, elem = ty$f64, sizes = NULL, type = NULL) {
  if (!is.null(type)) shape <- type
  if (inherits(shape, "rmlir_type")) {
    tensor_type <- shape
  } else {
    if (is.null(shape)) stop("alloc_tensor() requires a shape or tensor type", call. = FALSE)
    if (!is.numeric(shape)) stop("alloc_tensor() shape must be numeric", call. = FALSE)
    tensor_type <- ty$tensor_value(elem, shape = shape)
  }

  if (tensor_storage(tensor_type) != "tensor") {
    stop("alloc_tensor() requires a tensor_value type", call. = FALSE)
  }

  shape_vec <- tensor_type$shape %||% NA_integer_
  dyn_pos <- which(is.na(shape_vec))
  sizes_exprs <- list()
  if (length(dyn_pos) > 0) {
    if (is.null(sizes)) stop("alloc_tensor() requires sizes for dynamic dims", call. = FALSE)
    if (!is.list(sizes)) sizes <- as.list(sizes)
    if (length(sizes) != length(dyn_pos)) {
      stop("alloc_tensor() sizes must match number of dynamic dims", call. = FALSE)
    }
    sizes_exprs <- lapply(sizes, function(x) coerce_expr(x, ty$index))
  }

  ast <- list(
    type = "AllocTensor",
    tensorType = rmlir_type_to_ast(tensor_type),
    sizes = lapply(sizes_exprs, expr_ast)
  )
  rmlir_expr(ast, tensor_type)
}

clone <- function(x) {
  x_expr <- coerce_expr(x)
  require_tensor_value(x_expr, "clone()")
  ast <- list(type = "Clone", source = expr_ast(x_expr))
  rmlir_expr(ast, x_expr$type)
}

to_tensor <- function(x) {
  x_expr <- coerce_expr(x)
  require_memref(x_expr, "to_tensor()")
  out_type <- tensor_type_with_storage(x_expr$type, "tensor")
  ast <- list(type = "ToTensor", source = expr_ast(x_expr), tensorType = rmlir_type_to_ast(out_type))
  rmlir_expr(ast, out_type)
}

to_buffer <- function(x) {
  x_expr <- coerce_expr(x)
  require_tensor_value(x_expr, "to_buffer()")
  out_type <- tensor_type_with_storage(x_expr$type, "memref")
  ast <- list(type = "ToBuffer", source = expr_ast(x_expr), bufferType = rmlir_type_to_ast(out_type))
  rmlir_expr(ast, out_type)
}

coerce_host_values <- function(values, elem_type, feature) {
  if (!inherits(elem_type, "rmlir_type") || elem_type$kind != "prim") {
    stop(feature, " elem_type must be a primitive rmlir_type", call. = FALSE)
  }
  kind <- elem_type$name
  if (kind == "i32" || kind == "index") return(as.integer(values))
  if (kind == "f64") return(as.numeric(values))
  if (kind == "bool") return(as.logical(values))
  stop(feature, " unsupported element type: ", kind, call. = FALSE)
}

make_vector_buffer <- function(values, sym, elem_type = ty$i32) {
  rmlir_require_ctx("make_vector_buffer()")
  sym <- substitute(sym)
  if (!is.symbol(sym)) stop("make_vector_buffer() requires a symbol name", call. = FALSE)
  vals <- coerce_host_values(values, elem_type, "make_vector_buffer()")
  tensor <- alloc_tensor(type = ty$vector_value(elem_type, length(vals)))
  buf <- to_buffer(tensor)
  buf <- let(sym, buf)
  for (i in seq_along(vals)) {
    buf[i] <- const(vals[[i]], elem_type)
  }
  buf
}

make_matrix_buffer <- function(values, sym, elem_type = ty$i32) {
  rmlir_require_ctx("make_matrix_buffer()")
  sym <- substitute(sym)
  if (!is.symbol(sym)) stop("make_matrix_buffer() requires a symbol name", call. = FALSE)
  if (is.null(dim(values))) stop("make_matrix_buffer() requires a matrix", call. = FALSE)
  nrow_vals <- nrow(values)
  ncol_vals <- ncol(values)
  vals <- coerce_host_values(values, elem_type, "make_matrix_buffer()")
  vals <- matrix(vals, nrow = nrow_vals, ncol = ncol_vals)
  tensor <- alloc_tensor(type = ty$matrix_value(elem_type, nrow_vals, ncol_vals))
  buf <- to_buffer(tensor)
  buf <- let(sym, buf)
  for (i in seq_len(nrow_vals)) {
    for (j in seq_len(ncol_vals)) {
      buf[i, j] <- const(vals[i, j], elem_type)
    }
  }
  buf
}

materialize_in_destination <- function(source, dest) {
  source_expr <- coerce_expr(source)
  dest_expr <- coerce_expr(dest)
  require_tensor_value(source_expr, "materialize_in_destination()")
  require_tensor_value(dest_expr, "materialize_in_destination()")
  if (!identical(source_expr$type$elem$name, dest_expr$type$elem$name)) {
    stop("materialize_in_destination() element types must match", call. = FALSE)
  }
  ast <- list(
    type = "MaterializeInDestination",
    source = expr_ast(source_expr),
    dest = expr_ast(dest_expr)
  )
  rmlir_expr(ast, dest_expr$type)
}

dealloc <- function(x) {
  ctx <- rmlir_require_ctx("dealloc()")
  x_expr <- coerce_expr(x)
  require_memref(x_expr, "dealloc()")
  ctx$emit(list(type = "Dealloc", target = expr_ast(x_expr)))
  invisible(NULL)
}

dealloc_tensor <- function(x) {
  ctx <- rmlir_require_ctx("dealloc_tensor()")
  x_expr <- coerce_expr(x)
  require_tensor_value(x_expr, "dealloc_tensor()")
  ctx$emit(list(type = "DeallocTensor", target = expr_ast(x_expr)))
  invisible(NULL)
}

store <- function(target, ..., value) {
  ctx <- rmlir_require_ctx("store()")
  idx <- index(target, ...)
  if (is.list(idx$ast) && identical(idx$ast$type, "Slice")) {
    stop("store() does not support slice targets; index a single element instead", call. = FALSE)
  }
  stmt <- list(type = "Store", target = idx$ast, value = expr_ast(value))
  ctx$emit(stmt)
  invisible(NULL)
}

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

rmlir_func <- function(name, params, body, ret = NULL) {
  stopifnot(inherits(params, "rmlir_params"))
  if (!is.function(body)) stop("body must be a function", call. = FALSE)
  structure(list(name = name, params = params, body = body, ret = ret), class = "rmlir_func")
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

compile_func_body <- function(func) {
  ctx <- rmlir_ctx()
  old <- .rmlirdsl_state$ctx
  .rmlirdsl_state$ctx <- ctx
  on.exit({ .rmlirdsl_state$ctx <- old }, add = TRUE)

  params <- func$params
  args <- list()
  for (nm in names(params)) {
    pty <- params[[nm]]
    expr <- rmlir_expr(list(type = "Var", name = nm), pty)
    args[[nm]] <- expr
    ctx$bind(nm, expr)
  }

  do.call(func$body, args)

  params_ast <- lapply(names(params), function(nm) list(name = nm, type = rmlir_type_to_ast(params[[nm]])))
  list(
    type = "Function",
    name = func$name,
    params = params_ast,
    returnType = if (is.null(func$ret)) NULL else rmlir_type_to_ast(func$ret),
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
  if (is.null(entry)) entry <- funcs[[1]]$name

  func <- NULL
  for (f in funcs) if (identical(f$name, entry)) func <- f
  if (is.null(func)) stop("entry function not found: ", entry, call. = FALSE)

  if (is.null(.rmlirdsl_state$runtime_dll)) runtime_load()
  json <- render_json(module, pretty = FALSE)
  compile_sym <- getNativeSymbolInfo("rdslmlir_compile", PACKAGE = .rmlirdsl_state$runtime_dll)
  handle <- .Call(compile_sym, json, entry)

  params <- func$params
  ret <- func$ret
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

rmlir <- new.env(parent = emptyenv())
rmlir$ty <- ty
rmlir$Params <- Params
rmlir$fn <- fn
rmlir$module <- module
rmlir$ret <- ret
rmlir$for_ <- for_
rmlir$const <- const
rmlir$var <- var
rmlir$make_vector_buffer <- make_vector_buffer
rmlir$make_matrix_buffer <- make_matrix_buffer
rmlir$render_json <- render_json
rmlir$compile_mlir <- compile_mlir
rmlir$execute_mlir <- execute_mlir
rmlir$runtime_load <- runtime_load
rmlir$runtime_compile <- runtime_compile

rmlir$internal <- new.env(parent = emptyenv())
rmlir$internal$let <- let
rmlir$internal$index <- index
rmlir$internal$slice <- slice
rmlir$internal$matmul <- matmul
rmlir$internal$alloc_tensor <- alloc_tensor
rmlir$internal$clone <- clone
rmlir$internal$to_tensor <- to_tensor
rmlir$internal$to_buffer <- to_buffer
rmlir$internal$materialize_in_destination <- materialize_in_destination
rmlir$internal$dealloc <- dealloc
rmlir$internal$dealloc_tensor <- dealloc_tensor
rmlir$internal$range <- rmlir_range
rmlir$internal$dim <- dim
rmlir$internal$store <- store

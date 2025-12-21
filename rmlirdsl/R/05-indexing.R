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

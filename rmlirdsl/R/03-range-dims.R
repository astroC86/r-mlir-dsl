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

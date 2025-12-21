alloc_tensor <- function(shape = NULL, elem = ty$f64, sizes = NULL, type = NULL) {
  if (!is.null(type)) shape <- type
  if (inherits(shape, "rmlir_type")) {
    tensor_type <- shape
  } else {
    if (is.null(shape)) stop("alloc_tensor() requires a shape or tensor type", call. = FALSE)
    if (!is.numeric(shape)) stop("alloc_tensor() shape must be numeric", call. = FALSE)
    tensor_type <- ty$tensor_val(elem, shape = shape)
  }

  if (tensor_storage(tensor_type) != "tensor") {
    stop("alloc_tensor() requires a tensor_val type", call. = FALSE)
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

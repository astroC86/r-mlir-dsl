rmlir_type <- function(kind, ...) {
  structure(list(kind = kind, ...), class = "rmlir_type")
}

ty <- new.env(parent = emptyenv())
ty$i32 <- rmlir_type("prim", name = "i32")
ty$f64 <- rmlir_type("prim", name = "f64")
ty$bool <- rmlir_type("prim", name = "bool")
ty$index <- rmlir_type("prim", name = "index")

make_tensor_type <- function(elem, shape, storage, feature) {
  if (!inherits(elem, "rmlir_type") || elem$kind != "prim") {
    stop(feature, " elem must be a primitive rmlir_type", call. = FALSE)
  }
  if (!is.null(shape)) {
    if (!is.numeric(shape)) stop(feature, " shape must be numeric or NULL", call. = FALSE)
    shape <- as.integer(shape)
  }
  rmlir_type("tensor", elem = elem, shape = shape, storage = storage)
}

ty$tensor_ref <- function(elem, shape = NULL) {
  make_tensor_type(elem, shape, "memref", "tensor_ref()")
}

ty$tensor_val <- function(elem, shape = NULL) {
  make_tensor_type(elem, shape, "tensor", "tensor_val()")
}

ty$tensor <- function(elem, shape = NULL) {
  ty$tensor_ref(elem, shape)
}

ty$tensor_value <- function(elem, shape = NULL) {
  ty$tensor_val(elem, shape)
}

ty$mut_tensor <- function(elem, shape = NULL) {
  ty$tensor_ref(elem, shape)
}

ty$const_tensor <- function(elem, shape = NULL) {
  ty$tensor_val(elem, shape)
}

ty$vector_ref <- function(elem, n = NULL) {
  if (is.null(n)) return(ty$tensor_ref(elem, shape = NA_integer_))
  ty$tensor_ref(elem, shape = as.integer(n))
}

ty$vector_val <- function(elem, n = NULL) {
  if (is.null(n)) return(ty$tensor_val(elem, shape = NA_integer_))
  ty$tensor_val(elem, shape = as.integer(n))
}

ty$vector <- function(elem, n = NULL) {
  ty$vector_ref(elem, n)
}

ty$vector_value <- function(elem, n = NULL) {
  ty$vector_val(elem, n)
}

ty$mut_vector <- function(elem, n = NULL) {
  ty$vector_ref(elem, n)
}

ty$const_vector <- function(elem, n = NULL) {
  ty$vector_val(elem, n)
}

ty$matrix_ref <- function(elem, nrow = NULL, ncol = NULL) {
  shape <- c(if (is.null(nrow)) NA_integer_ else as.integer(nrow),
             if (is.null(ncol)) NA_integer_ else as.integer(ncol))
  ty$tensor_ref(elem, shape = shape)
}

ty$matrix_val <- function(elem, nrow = NULL, ncol = NULL) {
  shape <- c(if (is.null(nrow)) NA_integer_ else as.integer(nrow),
             if (is.null(ncol)) NA_integer_ else as.integer(ncol))
  ty$tensor_val(elem, shape = shape)
}

ty$matrix <- function(elem, nrow = NULL, ncol = NULL) {
  ty$matrix_ref(elem, nrow, ncol)
}

ty$matrix_value <- function(elem, nrow = NULL, ncol = NULL) {
  ty$matrix_val(elem, nrow, ncol)
}

ty$mut_matrix <- function(elem, nrow = NULL, ncol = NULL) {
  ty$matrix_ref(elem, nrow, ncol)
}

ty$const_matrix <- function(elem, nrow = NULL, ncol = NULL) {
  ty$matrix_val(elem, nrow, ncol)
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
    stop(feature, " requires a tensor_val; use rmlir$to_tensor() first", call. = FALSE)
  }
}

tensor_type_with_storage <- function(type, storage) {
  if (!inherits(type, "rmlir_type") || type$kind != "tensor") {
    stop("expected tensor type", call. = FALSE)
  }
  rmlir_type("tensor", elem = type$elem, shape = type$shape, storage = storage)
}

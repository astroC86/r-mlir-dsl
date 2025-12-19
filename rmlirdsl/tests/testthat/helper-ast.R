library(rmlirdsl)

render_ast <- function(module) {
  jsonlite::fromJSON(rmlir$render_json(module), simplifyVector = FALSE)
}

first_function <- function(ast) {
  ast$functions[[1]]
}

find_nodes <- function(node, type) {
  out <- list()
  walk <- function(x) {
    if (is.list(x)) {
      if (!is.null(x$type) && identical(x$type, type)) {
        out[[length(out) + 1L]] <<- x
      }
      for (el in x) walk(el)
    }
  }
  walk(node)
  out
}

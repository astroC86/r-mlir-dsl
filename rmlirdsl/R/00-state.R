.rmlirdsl_state <- new.env(parent = emptyenv())
.rmlirdsl_state$ctx <- NULL
.rmlirdsl_state$runtime_dll <- NULL

`%||%` <- function(x, y) if (is.null(x)) y else x

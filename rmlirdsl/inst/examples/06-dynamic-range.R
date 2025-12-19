library(rmlirdsl)

slice_tail <- rmlir$fn("slice_tail", rmlir$Params(m = ty$matrix(ty$i32)), ret = ty$i32)(function(m) {
  n <- dim(m)[1]
  start <- n - 2L
  slice <- m[start:n, ]
  rmlir$ret(slice[1, 1])
})

mod <- rmlir$module(slice_tail)
slice_fn <- rmlir$runtime_compile(mod, entry = "slice_tail")

m_vals <- matrix(as.integer(1:12), nrow = 4, byrow = TRUE)
start <- nrow(m_vals) - 2L
expected <- as.integer(m_vals[start, 1])
result <- slice_fn(m_vals)
cat("expected:", expected, "runtime:", result, "\n")

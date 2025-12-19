library(rmlirdsl)

inc_inplace <- rmlir$fn("inc_inplace", rmlir$Params(x = ty$vector(ty$i32)), ret = ty$i32)(function(x) {
  n <- dim(x)[1]
  rmlir$for_(i, 1, n, body = {
    x[i] <- x[i] + 1L
  })
  rmlir$ret(x[1])
})

mod <- rmlir$module(inc_inplace)
inc_fn <- rmlir$runtime_compile(mod, entry = "inc_inplace")

x_vals <- c(1L, 2L, 3L, 4L)
expected <- as.integer(x_vals[1] + 1L)
result <- inc_fn(x_vals)
cat("expected:", expected, "runtime:", result, "updated:", paste(x_vals, collapse = ", "), "\n")

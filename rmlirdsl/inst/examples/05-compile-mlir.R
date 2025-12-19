library(rmlirdsl)

diag_sum <- rmlir$fn("diag_sum", rmlir$Params(m = ty$matrix(ty$i32)), ret = ty$i32)(function(m) {
  rmlir$ret(m[1, 1] + m[2, 2])
})

mod <- rmlir$module(diag_sum)
diag_fn <- rmlir$runtime_compile(mod, entry = "diag_sum")

m_vals <- matrix(c(2L, 1L, 3L, 4L), nrow = 2, byrow = TRUE)
expected <- as.integer(m_vals[1, 1] + m_vals[2, 2])
result <- diag_fn(m_vals)
cat("expected:", expected, "runtime:", result, "\n")

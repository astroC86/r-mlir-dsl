library(rmlirdsl)

matmul <- rmlir$fn("matmul", rmlir$Params(a = ty$matrix(ty$i32), b = ty$matrix(ty$i32)), ret = ty$i32)(function(a, b) {
  c <- a %*% b
  rmlir$ret(c[1, 1])
})

row_slice <- rmlir$fn("row_slice", rmlir$Params(m = ty$matrix(ty$i32)), ret = ty$i32)(function(m) {
  row <- m[2, ]
  rmlir$ret(row[1])
})

mod <- rmlir$module(matmul, row_slice)
matmul_fn <- rmlir$runtime_compile(mod, entry = "matmul")
row_slice_fn <- rmlir$runtime_compile(mod, entry = "row_slice")

a_vals <- matrix(c(1L, 2L, 3L, 4L), nrow = 2, byrow = TRUE)
b_vals <- matrix(c(5L, 6L, 7L, 8L), nrow = 2, byrow = TRUE)

matmul_expected <- as.integer((a_vals %*% b_vals)[1, 1])
matmul_result <- matmul_fn(a_vals, b_vals)
cat("matmul expected:", matmul_expected, "runtime:", matmul_result, "\n")

row_expected <- as.integer(a_vals[2, 1])
row_result <- row_slice_fn(a_vals)
cat("row_slice expected:", row_expected, "runtime:", row_result, "\n")

library(rmlirdsl)

vec_ops <- rmlir$fn("vec_ops", rmlir$Params(x = ty$vector(ty$i32)), ret = ty$i32)(function(x) {
  slice <- x[2:4]
  sum_slice <- slice[1] + slice[2] + slice[3]
  head <- x[1]
  rmlir$ret(head + sum_slice)
})

vec_head <- rmlir$fn("vec_head", rmlir$Params(x = ty$vector(ty$i32)), ret = ty$i32)(function(x) {
  rmlir$ret(x[1])
})

mod <- rmlir$module(vec_ops, vec_head)
vec_ops_fn <- rmlir$runtime_compile(mod, entry = "vec_ops")
vec_head_fn <- rmlir$runtime_compile(mod, entry = "vec_head")

values <- c(10L, 20L, 30L, 40L, 50L, 60L, 70L, 80L)

expected_ops <- as.integer(values[1] + sum(values[2:4]))
result_ops <- vec_ops_fn(values)
cat("vec_ops expected:", expected_ops, "runtime:", result_ops, "\n")

expected_head <- as.integer(values[1])
result_head <- vec_head_fn(values)
cat("vec_head expected:", expected_head, "runtime:", result_head, "\n")

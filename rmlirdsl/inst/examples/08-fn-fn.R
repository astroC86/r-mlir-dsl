library(rmlirdsl)

inc <- rmlir$fn("inc", rmlir$Params(x = ty$i32), ret = ty$i32)(function(x) {
  rmlir$ret(x + 1)
})

vec_ops <- rmlir$fn("vec_ops", rmlir$Params(x = ty$vector(ty$i32)), ret = ty$i32)(function(x) {
  slice <- x[2:4]
  sum_slice <- inc(slice[1]) + inc(slice[2]) + inc(slice[3])
  head <- x[1]
  rmlir$ret(head + sum_slice)
})

mod        <- rmlir$module(vec_ops, inc)
inc_fn     <- rmlir$runtime_compile(mod, entry = "inc")
vec_ops_fn <- rmlir$runtime_compile(mod, entry = "vec_ops")

values <- c(10L, 20L, 30L, 40L, 50L, 60L, 70L, 80L)
result_ops <- vec_ops_fn(values)

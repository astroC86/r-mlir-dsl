library(rmlirdsl)

addmul <- rmlir$fn("addmul", rmlir$Params(x = ty$vector(ty$i32)), ret = ty$i32)(function(x) {
  z <- x[1] + x[2]
  rmlir$ret(z * 2L)
})

mod <- rmlir$module(addmul)
fn <- rmlir$runtime_compile(mod, entry = "addmul")

vals <- c(3L, 5L)
expected <- as.integer((vals[1] + vals[2]) * 2L)
result <- fn(vals)
cat("expected:", expected, "runtime:", result, "\n")

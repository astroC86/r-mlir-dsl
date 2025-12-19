library(rmlirdsl)
Sys.setenv(OMP_NUM_THREADS=2)
Sys.setenv(OMP_DISPLAY_ENV=TRUE)
set_inplace <- rmlir$fn("set_inplace", rmlir$Params(x = ty$vector(ty$i32)), ret = NULL)(function(x) {
  n <- dim(x)[1]
  rmlir$for_(i, 1, n, parallel=TRUE, target="openmp", body = {
    x[i] <- x[i] + 1L
  })
})

mod <- rmlir$module(set_inplace)
set_fn <- rmlir$runtime_compile(mod, entry = "set_inplace")
x_vals <- rep(0L, 1000000000)
nu <- set_fn(x_vals)

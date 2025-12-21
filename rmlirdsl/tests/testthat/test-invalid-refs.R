library(testthat)


skip_if_not_installed("codetools")


test_that("Sys.Sleep inside rmlir$fn is rejected", {
  bad <- rmlir$fn("bad_sleep", rmlir$Params(n = ty$index), ret = ty$i32)(function(n) {
    rmlir$for_(i, const(1, ty$index), n, body = {
      Sys.Sleep(0)
    })
    rmlir$ret(const(0L, ty$i32))
  })

  expect_error(render_ast(rmlir$module(bad)), "unsupported call")
})


test_that("unsupported R calls are rejected", {
  bad_simple <- rmlir$fn("bad_sum", rmlir$Params(x = ty$f64), ret = ty$f64)(function(x) {
    rmlir$ret(sum(x))
  })

  bad_namespace <- rmlir$fn("bad_stats", rmlir$Params(n = ty$index), ret = ty$f64)(function(n) {
    rmlir$ret(stats::rnorm(n))
  })

  bad_base <- rmlir$fn("bad_base", rmlir$Params(x = ty$vector_ref(ty$f64)), ret = ty$f64)(function(x) {
    total <- base::sum(x)
    rmlir$ret(total)
  })

  expect_error(render_ast(rmlir$module(bad_simple)), "unsupported call")
  expect_error(render_ast(rmlir$module(bad_namespace)), "unsupported call")
  expect_error(render_ast(rmlir$module(bad_base)), "unsupported call")
})


test_that("global references are rejected", {
  g <- 3L
  bad <- rmlir$fn("bad_global", rmlir$Params(x = ty$i32), ret = ty$i32)(function(x) {
    rmlir$ret(x + g)
  })

  expect_error(render_ast(rmlir$module(bad)), "unsupported global reference")
})

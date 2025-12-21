library(testthat)


skip_if_not_installed("codetools")


test_that("math functions are allowed in rmlir$fn bodies", {
  fn <- rmlir$fn("math_fns", rmlir$Params(x = ty$f64, y = ty$f64), ret = ty$f64)(function(x, y) {
    rmlir$ret(
      abs(x) + sqrt(x) + exp(x) + log(x) + log(x, base = 2) + log10(x) + log2(x) + sign(x) +
        sin(x) + cos(x) + tan(x) + asin(x) + acos(x) + atan(x) + atan2(x, y) +
        sinh(x) + cosh(x) + tanh(x) + asinh(x) + acosh(x) + atanh(x) +
        round(x) + round(x, 2) + floor(x) + ceiling(x) + trunc(x) + signif(x) + signif(x, 3)
    )
  })

  expect_no_error(render_ast(rmlir$module(fn)))
})

test_that("math functions on vectors are allowed in rmlir$fn bodies", {
  fn <- rmlir$fn("math_fns", rmlir$Params(x = ty$vector_ref(ty$f64), y = ty$vector_ref(ty$f64)), ret = ty$vector_ref(ty$f64))(function(x, y) {
    rmlir$ret(
      abs(x) + sqrt(x) + exp(x) + log(x) + log(x, base = 2) + log10(x) + log2(x) + sign(x) +
        sin(x) + cos(x) + tan(x) + asin(x) + acos(x) + atan(x) + atan2(x, y) +
        sinh(x) + cosh(x) + tanh(x) + asinh(x) + acosh(x) + atanh(x) +
        round(x) + round(x, 2) + floor(x) + ceiling(x) + trunc(x) + signif(x) + signif(x, 3)
    )
  })

  expect_no_error(render_ast(rmlir$module(fn)))
})

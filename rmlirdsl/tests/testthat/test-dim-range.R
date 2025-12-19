library(testthat)


test_that("dim() builds dynamic Dim ops when shape is unknown", {
  dims <- rmlir$fn("dims", rmlir$Params(m = ty$matrix(ty$f64, NA, NA)), ret = ty$index)(function(m) {
    n <- dim(m)[1]
    rmlir$ret(n)
  })

  fn <- first_function(render_ast(rmlir$module(dims)))
  expect_length(find_nodes(fn, "Dim"), 1)
})


test_that("dynamic ranges flow through slicing", {
  slice_dyn <- rmlir$fn("slice_dyn", rmlir$Params(m = ty$matrix(ty$f64, NA, NA)), ret = ty$matrix(ty$f64, NA, NA))(
    function(m) {
      start <- dim(m)[1] - 2
      rmlir$ret(m[start:dim(m)[1], ])
    }
  )

  fn <- first_function(render_ast(rmlir$module(slice_dyn)))
  expect_length(find_nodes(fn, "Slice"), 1)
  expect_true(length(find_nodes(fn, "Dim")) >= 1)
})

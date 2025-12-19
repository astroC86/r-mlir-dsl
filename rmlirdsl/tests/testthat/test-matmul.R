library(testthat)


test_that("%*% produces MatMul ops", {
  matmul <- rmlir$fn("matmul", rmlir$Params(a = ty$matrix(ty$f64, NA, NA), b = ty$matrix(ty$f64, NA, NA)),
    ret = ty$matrix(ty$f64, NA, NA))(
    function(a, b) {
      rmlir$ret(a %*% b)
    }
  )

  fn <- first_function(render_ast(rmlir$module(matmul)))
  expect_length(find_nodes(fn, "MatMul"), 1)
})

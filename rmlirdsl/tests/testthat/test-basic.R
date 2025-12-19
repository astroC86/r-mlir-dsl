library(testthat)


test_that("scalar expressions lower without explicit let", {
  addmul <- rmlir$fn("addmul", rmlir$Params(x = ty$f64, y = ty$f64), ret = ty$f64)(function(x, y) {
    z <- x + y
    rmlir$ret(z * 2.0)
  })

  ast <- render_ast(rmlir$module(addmul))
  fn <- first_function(ast)

  expect_equal(fn$name, "addmul")
  expect_equal(fn$returnType, "f64")
  expect_length(find_nodes(fn, "Assign"), 0)
  expect_length(find_nodes(fn, "Return"), 1)

  bin_ops <- find_nodes(fn, "Binary")
  expect_true(any(vapply(bin_ops, function(n) identical(n$op, "+"), logical(1))))
  expect_true(any(vapply(bin_ops, function(n) identical(n$op, "*"), logical(1))))
})


test_that("unary ops are encoded", {
  neg <- rmlir$fn("neg", rmlir$Params(x = ty$f64), ret = ty$f64)(function(x) {
    rmlir$ret(-x)
  })

  ast <- render_ast(rmlir$module(neg))
  fn <- first_function(ast)
  unary_ops <- find_nodes(fn, "Unary")
  expect_true(any(vapply(unary_ops, function(n) identical(n$op, "-"), logical(1))))
})

library(testthat)


test_that("vector indexing uses Index", {
  vec_index <- rmlir$fn("vec_index", rmlir$Params(x = ty$vector(ty$f64, 8)), ret = ty$f64)(function(x) {
    rmlir$ret(x[3])
  })

  fn <- first_function(render_ast(rmlir$module(vec_index)))
  expect_length(find_nodes(fn, "Index"), 1)
  expect_length(find_nodes(fn, "Slice"), 0)
})


test_that("vector slicing uses Slice", {
  vec_slice <- rmlir$fn("vec_slice", rmlir$Params(x = ty$vector(ty$f64, 8)), ret = ty$vector(ty$f64, 3))(
    function(x) {
      rmlir$ret(x[2:4])
    }
  )

  fn <- first_function(render_ast(rmlir$module(vec_slice)))
  slices <- find_nodes(fn, "Slice")
  expect_length(slices, 1)
  drop_dims <- unlist(slices[[1]]$dropDims)
  expect_equal(drop_dims, c(FALSE))
})


test_that("matrix indexing controls drop semantics", {
  row_slice <- rmlir$fn("row_slice", rmlir$Params(m = ty$matrix(ty$f64, 4, 5)), ret = ty$vector(ty$f64, 5))(
    function(m) {
      rmlir$ret(m[2, ])
    }
  )

  full_slice <- rmlir$fn("full_slice", rmlir$Params(m = ty$matrix(ty$f64, 4, 5)), ret = ty$matrix(ty$f64, 2, 3))(
    function(m) {
      rmlir$ret(m[1:2, 2:4, drop = FALSE])
    }
  )

  row_fn <- first_function(render_ast(rmlir$module(row_slice)))
  row_drop <- unlist(find_nodes(row_fn, "Slice")[[1]]$dropDims)
  expect_equal(row_drop, c(TRUE, FALSE))

  full_fn <- first_function(render_ast(rmlir$module(full_slice)))
  full_drop <- unlist(find_nodes(full_fn, "Slice")[[1]]$dropDims)
  expect_equal(full_drop, c(FALSE, FALSE))
})

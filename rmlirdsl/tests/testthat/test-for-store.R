library(testthat)


test_that("for_ emits a loop with stores", {
  inc <- rmlir$fn("inc", rmlir$Params(x = ty$vector(ty$f64, NA), out = ty$vector(ty$f64, NA)))(
    function(x, out) {
      n <- dim(x)[1]
      rmlir$for_(i, 1, n, body = {
        out[i] <- x[i] + 1
      })
      rmlir$ret()
    }
  )

  fn <- first_function(render_ast(rmlir$module(inc)))
  loops <- find_nodes(fn, "For")
  expect_length(loops, 1)
  expect_identical(loops[[1]]$parallel, FALSE)
  expect_length(find_nodes(loops[[1]], "Store"), 1)
})


test_that("parallel loops carry target annotations", {
  inc_par <- rmlir$fn("inc_par", rmlir$Params(x = ty$vector(ty$f64, NA), out = ty$vector(ty$f64, NA)))(
    function(x, out) {
      n <- dim(x)[1]
      rmlir$for_(i, 1, n, parallel = TRUE, target = "gpu", body = {
        out[i] <- x[i] + 1
      })
      rmlir$ret()
    }
  )

  fn <- first_function(render_ast(rmlir$module(inc_par)))
  loops <- find_nodes(fn, "For")
  expect_length(loops, 1)
  expect_identical(loops[[1]]$parallel, TRUE)
  expect_identical(loops[[1]]$target, "gpu")
})

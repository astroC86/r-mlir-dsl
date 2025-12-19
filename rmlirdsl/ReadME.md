rmlirdsl

rmlirdsl is a lightweight R‑centric DSL that emits a JSON AST from idiomatic R
expressions, lowers it to an MLIR dialect, and then compiles/executed the lowered
module on CPU (with OpenMP  support for `parallel = TRUE` loops). The project ships
a runtime that exposes the MLIR-compiled functions through `.Call`, so you can
pass and return native R vectors/matrices with very little ceremony.

## Quick example

```R
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
```

## Features

- DSL built on top of messages of tensors / vectors with `rmlir$fn`, `ty$vector`,
  `ty$matrix`, and optional `parallel = TRUE` loops that map to `scf.parallel`.
- MLIR dialect + lowering stack that emits Linalg/SCF/OpenMP loops and buffers.
- Runtime shared library exposes `rdslmlir_compile` / `rdslmlir_call`, so you can
  compile MLIR modules once and reuse them via `rmlir$runtime_compile()`.
- Examples in `rmlirdsl/inst/examples/` that cover scalar kernels, indexing,
  matmul, OpenMP loops, loop stores, and dynamic ranges.

## Build & run

```bash
cmake -S . -B build -DMLIR_DIR=$MLIR_DIR
cmake --build build -j4
```

Run the R tests/examples by pointing `R` at the shared runtime:

```R
library(rmlirdsl)
dyn.load("build/runtime/librdslmlir_runtime.so")
```

Then use `rmlir$runtime_compile()` to execute compiled functions. If you enable
`parallel = TRUE` loops, set `OMP_NUM_THREADS` before calling `runtime_compile`
to control CPU concurrency.

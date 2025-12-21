rmlir <- new.env(parent = emptyenv())
rmlir$ty <- ty
rmlir$Params <- Params
rmlir$fn <- fn
rmlir$module <- module
rmlir$ret <- ret
rmlir$for_ <- for_
rmlir$const <- const
rmlir$var <- var
rmlir$runtime_compile <- runtime_compile

rmlir$internal <- new.env(parent = emptyenv())
rmlir$internal$let <- let
rmlir$internal$index <- index
rmlir$internal$slice <- slice
rmlir$internal$matmul <- matmul
rmlir$internal$alloc_tensor <- alloc_tensor
rmlir$internal$clone <- clone
rmlir$internal$to_tensor <- to_tensor
rmlir$internal$to_buffer <- to_buffer
rmlir$internal$materialize_in_destination <- materialize_in_destination
rmlir$internal$dealloc <- dealloc
rmlir$internal$dealloc_tensor <- dealloc_tensor
rmlir$internal$range <- rmlir_range
rmlir$internal$dim <- dim
rmlir$internal$store <- store
rmlir$internal$render_json <- render_json
rmlir$internal$compile_mlir <- compile_mlir
rmlir$internal$execute_mlir <- execute_mlir
rmlir$internal$runtime_load <- runtime_load

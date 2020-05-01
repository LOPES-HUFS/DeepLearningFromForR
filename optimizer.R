sgd.update <- function(params, grads, lr = 0.01){
  for(i in names(params)){params[[i]] <- params[[i]] - (grads[[i]]*lr)}
  return(params) 
}

optimizer <- list(c(NULL), c(NULL), c(NULL))
names(optimizer) <- c("Momentum", "AdaGrad", "Adam")

momentum.update <- function(params, grad,v, lr = 0.01,momentum=0.9){
  if (is.null(v) == TRUE) {
    v <- rep(list(NA),NROW(params))
    names(v) <- names(params)
    for(i in 1:NROW(params)){
      v[[i]] <- matrix(0,dim(params[[i]])[1],dim(params[[i]])[2])
    }
  }
  for(i in 1:NROW(params)){
    optimizer$Momentum[[i]] <<- momentum*v[[i]] - lr*grad[[i]]
    params[[i]] <- params[[i]]+v[[i]]}
  names(optimizer$Momentum) <- names(params)
  return(params)
}
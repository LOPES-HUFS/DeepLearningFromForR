sgd.update <- function(network, grads, lr = 0.01){
  for(i in names(network)){network[[i]] <- network[[i]] - (grads[[i]]*lr)}
  return(network)
}

optimizer <- list(Momentum=NULL, AdaGrad=NULL, Adam=list("iter"=0,"m"=NULL,"v"=NULL))

momentum.update <- function(network, grad,v, lr = 0.01, momentum=0.9){
  if (is.null(v) == TRUE) {
    v <- rep(list(NA),NROW(network))
    names(v) <- names(network)
    for(i in 1:NROW(network)){
      v[[i]] <- matrix(0,dim(network[[i]])[1],dim(network[[i]])[2])
    }
  }
  for(i in 1:NROW(network)){
    optimizer$Momentum[[i]] <<- v[[i]]*momentum - (lr*grad[[i]])

    network[[i]] <- network[[i]]+optimizer$Momentum[[i]]}
  names(optimizer$Momentum) <- names(network)
  return(network)
}

AdaGrad.update <- function(network,grad,h,lr=0.01){
  if (is.null(h) == TRUE) {
    h <- rep(list(NA),NROW(network))
    names(h) <- names(network)
    for(i in 1:NROW(network)){
      h[[i]] <- matrix(0,dim(network[[i]])[1],dim(network[[i]])[2])
    }
  }

  for(i in 1:NROW(network)){
    optimizer$AdaGrad[[i]]  <<- h[[i]] + grad[[i]]^2
    network[[i]] <- network[[i]] - (lr*grad[[i]] / (sqrt(optimizer$AdaGrad[[i]])+1e-7))
  }
  names(optimizer$AdaGrad) <- names(network)
  return(network)
}

Adam.update <- function(network,grads,iter,m,v,lr=0.001,beta1=0.9,beta2=0.999){
  if(is.null(m) == TRUE){
    m <- rep(list(NA),NROW(network))
    v <- rep(list(NA),NROW(network))
    names(m) <- names(network)
    names(v) <- names(network)
    for(i in 1:NROW(network)){
      m[[i]] <- matrix(0,dim(network[[i]])[1],dim(network[[i]])[2])
      v[[i]] <- matrix(0,dim(network[[i]])[1],dim(network[[i]])[2])
    }
  }
  optimizer$Adam$iter <<- iter+1
  lr_t <- (lr*sqrt(1.0 - beta2^optimizer$Adam$iter))/ (1.0 - beta1^optimizer$Adam$iter)
  temp_m_list <- rep(list(NA),NROW(network))
  temp_v_list <- rep(list(NA),NROW(network))
  for(i in 1:NROW(network)){
    temp_m_list[[i]] <-  m[[i]] + (1 - beta1) * (grads[[i]] - m[[i]])
    temp_v_list[[i]] <-  v[[i]] + (1 - beta2) * (grads[[i]]^2 - v[[i]])
    network[[i]] <- network[[i]] - (lr_t * temp_m_list[[i]]/ (sqrt(temp_v_list[[i]]) + 1e-7))
  }
  optimizer$Adam$m <- temp_m_list
  optimizer$Adam$v <- temp_v_list
  return(network)
}

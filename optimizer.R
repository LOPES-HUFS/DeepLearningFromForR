sgd.update <- function(params, grads, lr = 0.01){
  for(i in names(params)){params[[i]] <- params[[i]] - (grads[[i]]*lr)}
  return(params) 
}

optimizer <- list(Momentum=NULL, AdaGrad=NULL, Adam=list("iter"=0,"m"=NULL,"v"=NULL))

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

AdaGrad.update <- function(params,grad,h,lr=0.01){
  if (is.null(h) == TRUE) {
    h <- rep(list(NA),NROW(params))
    names(h) <- names(params)
    for(i in 1:NROW(params)){
      h[[i]] <- matrix(0,dim(params[[i]])[1],dim(params[[i]])[2])
    }
  }
  
  for(i in 1:NROW(params)){
    optimizer$AdaGrad[[i]]  <<- h[[i]] + grad[[i]]^2
    params[[i]] <- params[[i]] - (lr*grad[[i]] / (sqrt(optimizer$AdaGrad[[i]])+1e-7))
  }
  names(optimizer$AdaGrad) <- names(params)
  return(params)
}

Adam.update <- function(params,grads,iter,m,v,lr=0.001,beta1=0.9,beta2=0.999){
  if(is.null(m) == TRUE){
    m <- rep(list(NA),NROW(params))
    v <- rep(list(NA),NROW(params))
    names(m) <- names(params)
    names(v) <- names(params)
    for(i in 1:NROW(params)){
      m[[i]] <- matrix(0,dim(params[[i]])[1],dim(params[[i]])[2])
      v[[i]] <- matrix(0,dim(params[[i]])[1],dim(params[[i]])[2])
    }
  }
  optimizer$Adam$iter <<- iter+1
  lr_t <- (lr*sqrt(1.0 - beta2^optimizer$Adam$iter))/ (1.0 - beta1^optimizer$Adam$iter) 
  temp_m_list <- rep(list(NA),NROW(params))
  temp_v_list <- rep(list(NA),NROW(params))
  for(i in 1:NROW(params)){
    temp_m_list[[i]] <-  m[[i]] + (1 - beta1) * (grads[[i]] - m[[i]])
    temp_v_list[[i]] <-  v[[i]] + (1 - beta2) * (grads[[i]]^2 - v[[i]])
    params[[i]] <- params[[i]] - (lr_t * temp_m_list[[i]]/ (sqrt(temp_v_list[[i]]) + 1e-7))
  }
  optimizer$Adam$m <- temp_m_list
  optimizer$Adam$v <- temp_v_list
  return(params)
}
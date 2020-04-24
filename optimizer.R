sgd.update <- function(params,grads,lr=0.1){
  sgd <- function(params,grads,weight,lr=0.1){
    params[[weight]] <- params[[weight]] - (grads[[weight]]*lr)
    return(params[[weight]])
  }
  params_temp <- lapply(names(params),FUN = function(x){sgd(params = params,grads = grad,weight = x)})
  names(params_temp) <- names(params)
  params <- params_temp
  return(params) 
}

cross_entropy_error <- function(y, t){
  delta <- 1e-7
  batchsize <- dim(y)[1]
  return(-sum(t * log(y + delta))/batchsize)
}

softmax <- function(a){
  exp_a <- exp(a - apply(a,1,max))
  return(sweep(exp_a,1,rowSums(exp_a),"/"))
}

simplemodel  <- function(){
  W <<- matrix(c(0.47355232, 0.85557411, 0.9977393, 0.03563661,0.84668094,0.69422093), nrow = 2)
}
simplemodel()
W

simplemodel.forward <- function(x) {
  return(x %*% W)
}

numerical_gradient_simplemodel <- function(f){
  h <- 1e-4
  vec <- vector()
  
  for(i in 1:length(W)){
    W[i] <<- (W[i] + h)
    fxh1 <- f(W)
    W[i] <<- (W[i] - (2*h))
    fxh2 <- f(W)
    vec <- c(vec, (fxh1 - fxh2) / (2*h))
    W[i] <<- W[i]+h
  }
  return(matrix(vec, nrow = nrow(W) ,ncol = ncol(W)))
}

x <- matrix(c(0.6,0.9),nrow = 1,ncol = 2)
x

p <- simplemodel.forward(x)
p   

t <- matrix(c(0,0,1),nrow = 1,ncol = 3)
t

loss <- function(W){cross_entropy_error(softmax(simplemodel.forward (x)),t)}
loss()
# 2층 신경망 구현을 위한 수치미분 함수입니다.
# 이 함수 코드에 대한 설명은 https://choosunsick.github.io/post/neural_network_5/ 에서 찾아보실 수 있습니다.

numerical_gradient_W1 <- function(f,x,t){
  h <- 1e-4
  vec <- matrix(0, nrow = nrow(W1) ,ncol = ncol(W1))
  for(i in 1:length(W1)){
    origin <- W1[i]
    W1[i] <<- (W1[i] + h)
    fxh1 <- f(x, t)
    W1[i] <<- (W1[i] - (2*h))
    fxh2 <- f(x, t)
    vec[i] <- (fxh1 - fxh2) / (2*h)
    W1[i] <<- origin
  }
  return(vec)
}
numerical_gradient_W2 <- function(f,x,t){
  h <- 1e-4
  vec <- matrix(0, nrow = nrow(W2) ,ncol = ncol(W2))
  for(i in 1:length(W2)){
    origin <- W2[i]
    W2[i] <<- (W2[i] + h)
    fxh1 <- f(x, t)
    W2[i] <<- (W2[i] - (2*h))
    fxh2 <- f(x, t)
    vec[i] <- (fxh1 - fxh2) / (2*h)
    W2[i] <<- origin
  }
  return(vec)
}
numerical_gradient_b1 <- function(f,x,t){
  h <- 1e-4
  vec <- matrix(0, nrow = nrow(b1) ,ncol = ncol(b1))
  for(i in 1:length(b1)){
    origin <- b1[i]
    b1[i] <<- (b1[i] + h)
    fxh1 <- f(x, t)
    b1[i] <<- (b1[i] - (2*h))
    fxh2 <- f(x, t)
    vec[i] <- (fxh1 - fxh2) / (2*h)
    b1[i] <<- origin
  }
  return(vec)
}

numerical_gradient_b2 <- function(f,x,t){
  h <- 1e-4
  vec <- matrix(0, nrow = nrow(b2) ,ncol = ncol(b2))
  for(i in 1:length(b2)){
    origin <- b2[i]
    b2[i] <<- (b2[i] + h)
    fxh1 <- f(x, t)
    b2[i] <<- (b2[i] - (2*h))
    fxh2 <- f(x, t)
    vec[i] <- (fxh1 - fxh2) / (2*h)
    b2[i] <<- origin
  }
  return(vec)
}

numerical_gradient <- function(f,x,t) {
  grads  <- list(W1 = numerical_gradient_W1(f,x,t), 
                 b1 = numerical_gradient_b1(f,x,t), 
                 W2 = numerical_gradient_W2(f,x,t), 
                 b2 = numerical_gradient_b2(f,x,t))
  return(grads)
}

numerical_gradient <- compiler::cmpfun(numerical_gradient)
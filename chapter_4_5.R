TwoLayerNet  <- function(input_size, hidden_size, output_size, weight_init_std = 0.01) {
  W1 <<- weight_init_std*matrix(rnorm(n = input_size*hidden_size), nrow = input_size, ncol = hidden_size)
  b1 <<- matrix(rep(0,hidden_size),nrow=1,ncol=hidden_size)
  W2 <<- weight_init_std*matrix(rnorm(n = hidden_size*output_size), nrow = hidden_size, ncol = output_size)
  b2 <<- matrix(rep(0,output_size),nrow=1,ncol=output_size)
  return(list(input_size, hidden_size, output_size,weight_init_std))
}

net <- TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

library(dslabs)

mnist<-read_mnist()
x_train<-mnist$train$images
t_train<-mnist$train$labels
x_test<-mnist$test$images
t_test<-mnist$test$labels

t_train_onehotlabel <- matrix(FALSE,nrow = 60000,ncol = 10)
t_index <- t_train+1
for(i in 1:NROW(t_train_onehotlabel)){
  t_train_onehotlabel[i, t_index[i]] <- TRUE
}

t_test_onehotlabel <- matrix(FALSE,nrow = 10000,ncol = 10)
t_index <- t_test+1
for(i in 1:NROW(t_test_onehotlabel)){
  t_test_onehotlabel[i, t_index[i]] <- TRUE
}

x_train_normalize <- x_train/255
x_test_normalize <- x_test/255

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}

softmax <- function(a){
  exp_a <- exp(a - apply(a,1,max))
  return(sweep(exp_a,1,rowSums(exp_a),"/"))
}

predict <- function(x){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  return(softmax(sweep((z1 %*% W2),2, b2,'+')))
}

cross_entropy_error <- function(y, t){
  delta <- 1e-7
  batchsize <- dim(y)[1]
  return(-sum(t * log(y + delta))/batchsize)
}

loss <-function(x,t){
  return(cross_entropy_error(predict(x),t))
}

loss_W1<-function(W1){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  y<- softmax(sweep((z1 %*% W2),2, b2,'+'))
  return(cross_entropy_error(y,t))
}

loss_W2<-function(W2){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  y<- softmax(sweep((z1 %*% W2),2, b2,'+'))
  return(cross_entropy_error(y,t))
}

loss_b1<-function(b1){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  y<- softmax(sweep((z1 %*% W2),2, b2,'+'))
  return(cross_entropy_error(y,t))
}

loss_b2<-function(b2){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  y<- softmax(sweep((z1 %*% W2),2, b2,'+'))
  return(cross_entropy_error(y,t))
}

accuracy <- function(x,t){
  y <- max.col(predict(x))
  t <- max.col(t)
  accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
  return(accuracy)
}

numerical_gradient_W1 <- function(f){
  vec <- vector()
  temp <- rep(0,length(W1))
  h <- 1e-4
  for(i in 1:length(W1)){
    temp[i] <- temp[i]+ h
    fxh1  <-  f(W1+temp)
    temp[i] <- (temp[i] - (2*h))
    fxh2  <- f(W1+temp)
    vec <- c(vec, (fxh1 - fxh2) / (2*h))
    temp[i] <- 0 
  }
  return(matrix(vec, nrow = nrow(W1) ,ncol = ncol(W1)))    
}

numerical_gradient_W2 <- function(f){
  vec <- vector()
  temp <- rep(0,length(W2))
  h <- 1e-4
  for(i in 1:length(W2)){
    temp[i] <- temp[i]+ h
    fxh1  <-  f(W2+temp)
    temp[i] <- (temp[i] - (2*h))
    fxh2  <- f(W2+temp)
    vec <- c(vec, (fxh1 - fxh2) / (2*h))
    temp[i] <- 0 
  }
  return(matrix(vec, nrow = nrow(W2) ,ncol = ncol(W2)))    
}

numerical_gradient_b1 <- function(f){
  vec <- vector()
  temp <- rep(0,length(b1))
  h <- 1e-4
  for(i in 1:length(b1)){
    temp[i] <- temp[i]+ h
    fxh1  <-  f(b1+temp)
    temp[i] <- (temp[i] - (2*h))
    fxh2  <- f(b1+temp)
    vec <- c(vec, (fxh1 - fxh2) / (2*h))
    temp[i] <- 0 
  }
  return(matrix(vec, nrow = nrow(b1) ,ncol = ncol(b1)))    
}

numerical_gradient_b2 <- function(f){
  vec <- vector()
  temp <- rep(0,length(b2))
  h <- 1e-4
  for(i in 1:length(b2)){
    temp[i] <- temp[i]+ h
    fxh1  <-  f(b2+temp)
    temp[i] <- (temp[i] - (2*h))
    fxh2  <- f(b2+temp)
    vec <- c(vec, (fxh1 - fxh2) / (2*h))
    temp[i] <- 0
  }
  return(matrix(vec, nrow = nrow(b2) ,ncol = ncol(b2)))    
}

numerical_gradient <- function(x, t) {
  grads  <- list(W1 = numerical_gradient_W1(loss_W1), 
                 b1 = numerical_gradient_b1(loss_W2), 
                 W2 = numerical_gradient_W2(loss_b1), 
                 b2 = numerical_gradient_b2(loss_b2))
  return(grads)
}
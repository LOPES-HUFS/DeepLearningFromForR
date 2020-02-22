TwoLayerNet  <- function(input_size, hidden_size, output_size, weight_init_std = 0.01) {
  W1 <<- weight_init_std*matrix(rnorm(n = input_size*hidden_size), nrow = input_size, ncol = hidden_size)
  b1 <<- matrix(rep(0,hidden_size),nrow=1,ncol=hidden_size)
  W2 <<- weight_init_std*matrix(rnorm(n = hidden_size*output_size), nrow = hidden_size, ncol = output_size)
  b2 <<- matrix(rep(0,output_size),nrow=1,ncol=output_size)
  return(list(input_size, hidden_size, output_size,weight_init_std))
}

net <- TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

library(dslabs)

mnist<-read_mnist()
x_train<-mnist$train$images
t_train<-mnist$train$labels
x_test<-mnist$test$images
t_test<-mnist$test$labels

making_one_hot_label <-function(t_label,nrow,ncol){
  data <- matrix(FALSE,nrow = nrow,ncol = ncol)
  t_index <- t_label+1
  for(i in 1:NROW(data)){
    data[i, t_index[i]] <- TRUE
  }
  return(data)
}
t_train_onehotlabel<-making_one_hot_label(t_train,60000,10)
t_test_onehotlabel<-making_one_hot_label(t_test,10000,10)

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

model.evaluate <- function(x,t){
  y <- max.col(predict(x))
  t <- max.col(t)
  accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
  return(accuracy)
}

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

numerical_gradient <- function(f,x, t) {
  grads  <- list(W1 = numerical_gradient_W1(f,x,t), 
                 b1 = numerical_gradient_b1(f,x,t), 
                 W2 = numerical_gradient_W2(f,x,t), 
                 b2 = numerical_gradient_b2(f,x,t))
  return(grads)
}

numerical_gradient <- compiler::cmpfun(numerical_gradient)
softmax <- compiler::cmpfun(softmax)

# trainning

learning_rate <- 0.1
iters_num <- 100
train_loss_list <- data.frame(lossvalue=rep(0,iters_num))
train_size <- dim(x_train_normalize)[1]
batch_size <- 100

for(i in 1:iters_num){
  batch_mask <- sample(train_size,batch_size)
  x <- x_train_normalize[batch_mask,]
  t <- t_train_onehotlabel[batch_mask,]
  grads <- numerical_gradient(loss,x,t)
  W1 <- W1 - (grads$W1 * learning_rate)
  W2 <- W2 - (grads$W2 * learning_rate)
  b1 <- b1 - (grads$b1 * learning_rate)
  b2 <- b2 - (grads$b2 * learning_rate)
  loss_value <- loss(x, t)
  train_loss_list[i,1] <- loss_value
  print(i)
}
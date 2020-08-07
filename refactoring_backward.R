#  rm(list=ls())
#  setwd('/Users/yejin/Sites/DeepLearningFromForR')
library(dslabs)

source("./layers.R")
source("./utils.R")
source("./model.R")

TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}

init <- function(){
  mnist_data <- get_data()
  #손글씨 데이터
  x_train_normalize <<- mnist_data$x_train 
  x_test_normalize <<- mnist_data$x_test
  #정답 레이블
  t_train_onehotlabel <<- making_one_hot_label(mnist_data$t_train,60000, 10)
  t_test_onehotlabel <<- making_one_hot_label(mnist_data$t_test,10000, 10)
}

model.forward <- function(network, x){
  affine_1 <- Affine.forward(network$W1, network$b1, x)
  relu_1 <- Relu.forward(affine_1$out)
  affine_2 <- Affine.forward(network$W2, network$b2, relu_1$out)
  softmax <- softmax(affine_2$out)
  
  return(list(
    affine_1 = affine_1,
    relu_1 = relu_1,
    affine_2 = affine_2,
    softmax = softmax
  ))
}

model.backward <- function(network, x, t) {
  # 순전파
  d_forward <- model.forward(network, x)
  
  # 역전파
  dout <- 1
  last_backward <- SoftmaxWithLoss.backward(d_forward$softmax, t, dout)
  affine_2_backward <- Affine.backward(d_forward$affine_2, last_backward$dx)
  relu_1_backward <- Relu.backward(d_forward$relu_1, affine_2_backward$dx)
  affine_1_backward <- Affine.backward(d_forward$affine_1, relu_1_backward$dx)
  
  return(list(
    W1 = affine_1_backward$dW, 
    b1 = affine_1_backward$db, 
    W2 = affine_2_backward$dW, 
    b2 = affine_2_backward$db
  ))
}

SoftmaxWithLoss.backward <- function(predict, t, dout=1){
  dx <- (predict - t) / dim(predict)[1]
  return(list(dx = dx))
}

loss <- function(model.forward, network, x, t){
  temp <- model.forward(network, x)
  y <- temp$affine_2$out
  last_layer.forward <- SoftmaxWithLoss.forward(y,t) 
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}

model.train <- function(batch_size, iters_num, learning_rate, optimizer_name, debug=FALSE){
  train_size <- dim(x_train_normalize)[1]
  iter_per_epoch <- max(train_size / batch_size)
  train_loss_list <- data.frame(loss_value = 0)
  test_acc <- data.frame(acc = 0)
  network <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
  for(i in 1:iters_num){
    batch_mask <- sample(train_size ,batch_size)
    x_batch <- x_train_normalize[batch_mask,]
    t_batch <- t_train_onehotlabel[batch_mask,]
    
    gradient <- model.backward(network, x_batch, t_batch)
    network <- get_optimizer(network, gradient, optimizer_name)
    loss_value <- loss(model.forward=model.forward,network,x_batch, t_batch)$loss
    train_loss_list <- rbind(train_loss_list,loss_value)
    if(debug){
      if(i %% iter_per_epoch == 0){
        test_acc <- rbind(test_acc,model.evaluate(model.forward, network, x_test_normalize, t_test_onehotlabel))
      }
    }
  }
  return(list(network=network,train_loss_list=train_loss_list, test_accuracy=test_acc))
}

init()
model.train(100, 12000, 0.1, "adam", TRUE)

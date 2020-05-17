rm(list=ls())
setwd('/Users/yejin/Sites/DeepLearningFromForR')
library(dslabs)

source("./functions.R")
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

model.forward <- function(network,x){
  Affine_1 <- Affine.forward(network$W1, network$b1, x)
  Relu_1 <- Relu.forward(Affine_1$out)
  Affine_2 <- Affine.forward(network$W2, network$b2, Relu_1$out)
  return(list(x = Affine_2$out, Affine_1.forward = Affine_1, Affine_2.forward = Affine_2, Relu_1.forward = Relu_1))
}

loss <- function(network,model.forward, x, t){
  temp <- model.forward(network,x)
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}

model.backward <- function(network, model.forward, x, t) {
  # 순전파
  d_forward <- loss(network,model.forward, x, t)
  # 역전파
  dout <- 1
  last.backward <- SoftmaxWithLoss.backward(d_forward$softmax, dout)
  Affine_2.backward <- Affine.backward(d_forward$predict$Affine_2.forward, dout  =  last.backward$dx)
  Relu_1.backward <- Relu.backward(d_forward$predict$Relu_1.forward, dout  =  Affine_2.backward$dx)
  Affine_1.backward <- Affine.backward(d_forward$predict$Affine_1.forward, dout  =  Relu_1.backward$dx)
  grads  <- list(W1  =  Affine_1.backward$dW, b1  =  Affine_1.backward$db, W2  =  Affine_2.backward$dW, b2  =  Affine_2.backward$db)
  return(grads)
}

train_model <- function(batch_size, iters_num, learning_rate, optimizer_name, debug=FALSE){
  #seperate train, test data
  train_size <- dim(x_train_normalize)[1]

  iter_per_epoch <- max(train_size / batch_size)
  network <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
  for(i in 1:iters_num){
      batch_mask <- sample(train_size ,batch_size)
      x_batch <- x_train_normalize[batch_mask,]
      t_batch <- t_train_onehotlabel[batch_mask,]

      grad <- model.backward(network, model.forward=model.forward, x_batch, t_batch)
      #update weights and biases using SGD
      network <- get_optimizer(network,grad,optimizer_name)

      if(debug == TRUE){
          if(i %% iter_per_epoch == 0){
              train_acc <- model.evaluate(network,model.forward, x_train_normalize, t_train_onehotlabel)
              test_acc <- model.evaluate(network,model.forward, x_test_normalize, t_test_onehotlabel)
              print(c(train_acc, test_acc))
          }
      }
  }

  train_accuracy = model.evaluate(network,model.forward, x_train_normalize, t_train_onehotlabel)
  test_accuracy = model.evaluate(network,model.forward, x_test_normalize, t_test_onehotlabel)
  return(c(train_accuracy, test_accuracy))
}

init()
train_model(100, 10000, 0.1, "SGD", TRUE)



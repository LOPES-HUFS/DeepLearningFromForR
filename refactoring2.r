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

model.init <- function(train_data, test_data, train_answer, test_answer){
    x_train_normalize <<- train_data
    x_test_normalize <<- test_data
    t_train_onehotlabel <<- train_answer
    t_test_onehotlabel <<- test_answer
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

loss <- function(predict, network, x, t){
  predict <- model.forward(network, x)
  y <- predict$affine_2$out
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict = predict))
}


model.predict<-function(network, model.forward, x){
    return(model.forward(network, x)$softmax)
}

model.evaluate <- function(predict,x,t){
  answer <- max.col(t)
  accuracy <- (sum(ifelse(predict == answer,1,0))) / dim(x)[1]
  return(accuracy)
}

model.train <- function(network, model.forward, model.backward, optimizer="SGD", batch_size=100, iters_num=10000, learning_rate=0.01, debug=FALSE){
    print("start")
    train_size <- dim(x_train_normalize)[1]
    iter_per_epoch <- max(train_size / batch_size)

    train_loss_list <- data.frame(lossvalue  =  0)
    train_acc_list <- data.frame(train_acc  =  0)
    test_acc_list <- data.frame(test_acc  =  0)
    for(i in 1:iters_num){
        print(i)
        batch_mask <- sample(train_size, batch_size)
        x_batch <- x_train_normalize[batch_mask,]
        t_batch <- t_train_onehotlabel[batch_mask,]

        gradient <- model.backward(network, x_batch, t_batch)
        network <- get_optimizer(network, gradient, optimizer)
        if(debug){
            predict <- model.predict(network, model.forward, x_train_normalize)
            loss_value <- loss(predict, network, x_batch, t_batch)$loss
            train_loss_list <- rbind(train_loss_list,loss_value)
            if(i %% iter_per_epoch == 0){
                train_acc <- model.evaluate(predict, x_train_normalize, t_train_onehotlabel)
                test_acc <- model.evaluate(predict, x_test_normalize, t_test_onehotlabel)
                train_acc_list <- rbind(train_acc_list,train_acc)
                test_acc_list <- rbind(test_acc_list,test_acc)

                print(c(train_acc, test_acc))
            }
        }
    }

    train_predict <- model.predict(network, model.forward, x_train_normalize)
    test_predict <- model.predict(network, model.forward, x_test_normalize)
    train_accuracy = model.evaluate(train_predict, x_train_normalize, t_train_onehotlabel)
    test_accuracy = model.evaluate(test_predict, x_test_normalize, t_test_onehotlabel)
    print(train_accuracy, test_accuracy)

    if(debug){
        return(
            network=network,
            loss=train_loss_list,
            accuracy=merge(train_acc_list, test_acc_list)
        )
    }
    return(network=network)
}

mnist_data <- get_data()
train_answer <- making_one_hot_label(mnist_data$t_train,60000, 10)
test_answer <- making_one_hot_label(mnist_data$t_test,10000, 10)
model.init(mnist_data$x_train, mnist_data$x_test, train_answer, test_answer)
network <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
model.train(network, model.forward, model.backward, optimizer="SGD", batch_size=100, iters_num=10000, learning_rate=0.01, debug=TRUE)

# 아래와 같이 model.forward를 만들어
#
# model.forward <- function(x){
#   Affine_1 <- Affine.forward(network$W1, network$b1, x)
#   Relu_1 <- Relu.forward(Affine_1$out)
#   Affine_2 <- Affine.forward(network$W2, network$b2, Relu_1$out)
#   return(list(x = Affine_2$out, Affine_1.forward = Affine_1, Affine_2.forward = Affine_2, Relu_1.forward = Relu_1))
# }
#
# 모델을 평가할 때, 아래와 같이 사용하면 된다.
#
# model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
# model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)

source("./optimizer.R")

model.backward <- function(){
    return(FALSE)
}

model.forward <- function(){
    return(FALSE)
}

model.evaluate <- function(func,network,x,t){
  model <- func(network, x)
  predict <- max.col(model$softmax)
  answer <- max.col(t)
  accuracy <- (sum(ifelse(predict == answer,1,0))) / dim(x)[1]
  return(accuracy)
}

# 특정 숫자를 맞출 수 있는지 아래와 같이 사용하면 됩니다.

# model.predict(model.forward, x_train_normalize[2,])

model.predict <- function(model.forward, x){
    return(softmax(model.forward(x)))
}

model.train_set_validator <- function(x_train, y_train){
    return(dim(x_train)[2] != dim(x_test)[2])
}

model.train <- function(batch_size, iters_num, learning_rate, optimizer_name, debug=FALSE){
  train_size <- dim(x_train_normalize)[1]
  iter_per_epoch <- max(train_size / batch_size)

  network <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
  for(i in 1:iters_num){
      batch_mask <- sample(train_size ,batch_size)
      x_batch <- x_train_normalize[batch_mask,]
      t_batch <- t_train_onehotlabel[batch_mask,]
      
      gradient <- model.backward(network, x_batch, t_batch)
      network <- get_optimizer(network, gradient, optimizer_name)

      if(debug){
          if(i %% iter_per_epoch == 0){
              train_acc <- model.evaluate(model.forward, network, x_train_normalize, t_train_onehotlabel)
              test_acc <- model.evaluate(model.forward, network, x_test_normalize, t_test_onehotlabel)
              print(c(train_acc, test_acc))
          }
      }
  }

  train_accuracy = model.evaluate(model.forward, network, x_train_normalize, t_train_onehotlabel)
  test_accuracy = model.evaluate(model.forward, network, x_test_normalize, t_test_onehotlabel)
  return(c(train_accuracy, test_accuracy))
}
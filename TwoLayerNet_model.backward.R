# 2층 신경망 역전파 구현 모델입니다. 
# 역전파 예측과 손실함수 모델의 정확도를 계산하는 함수 입니다. 
# 관련 내용에 대한 설명은 https://choosunsick.github.io/post/neural_network_backward_4/ 에서 찾아보실 수 있습니다.

source("./functions.R")

model.backward <- function(x){
  Affine_1_layer <- Affine.forward(params$W1, params$b1, x)
  Relu_1_layer <- Relu.forward(Affine_1_layer$out)
  Affine_2_layer <- Affine.forward(params$W2, params$b2, Relu_1_layer$out)
  return(list(x = Affine_2_layer$out, Affine_1.forward = Affine_1_layer, Affine_2.forward = Affine_2_layer, Relu_1.forward = Relu_1_layer))
}

loss <- function(x, t){
  temp  <- model.backward(x) 
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}

model.evaluate.backward <- function(x,t){
  y <- max.col(model.backward(x)$x)
  t <- max.col(t)
  accuracy <- (sum(ifelse(y == t,1,0))) / dim(x)[1]
  return(accuracy)
}

predict.backward <- function(x){
  return(softmax(model.backward(x)$x))
}
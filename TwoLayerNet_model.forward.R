# 2층 신경망 순전파 구현 모델입니다. 
# 순전파 예측과 손실함수 모델의 정확도를 계산하는 함수 입니다. 
# 관련 내용에 대한 설명은 https://choosunsick.github.io/post/neural_network_5/ 에서 찾아보실 수 있습니다.

source("./functions.R")

model.forward <- function(x){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  return(softmax(sweep((z1 %*% W2),2, b2,'+')))
}

loss <-function(x,t){
  return(cross_entropy_error(model.forward(x),t))
}

model.evaluate.forward <- function(x,t){
  y <- max.col(model.forward(x))
  t <- max.col(t)
  accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
  return(accuracy)
}

predict.forward <- function(x){
  return(softmax(model.backward(x)))
}
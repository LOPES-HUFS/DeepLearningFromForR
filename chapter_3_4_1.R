# 간단한 신경망 모델을 만드는 스크립트입니다. 
# 이 스크립트의 코드에 대한 설명은 다음 링크에서 (https://choosunsick.github.io/post/neural_network_5/) 확인할 수 있습니다.
# 이 스크립트는 함수만 있는 것이 아니라 함수의 작동 결과 코드도 존재합니다. 
# 새롭게 정의하는 함수가 아닌 경우 `source()` 함수로 불러와 줍니다.

source("./functions.R")

# 간단한 신경망 모델의 초기값을 생성하는 함수입니다. 

simplemodel  <- function(){
  W <<- matrix(c(0.47355232, 0.85557411, 0.9977393, 0.03563661,0.84668094,0.69422093), nrow = 2)
}

simplemodel()
W

# 간단한 신경망 모델의 가중치를 계산하는 과정입니다.

simplemodel.forward <- function(x) {
  return(x %*% W)
}

# 간단한 신경망 모델에 맞추어 수치미분을 계산하는 함수입니다.  

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

dw <- numerical_gradient_simplemodel(loss)
dw
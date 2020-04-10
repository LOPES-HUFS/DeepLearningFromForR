# 이 스크립트는 오차역전파법을 적용한 학습과정 중 기울기검증을 구현한 것입니다. 
# 스크립트 내 코드의 자세한 설명은 다음 링크(https://choosunsick.github.io/post/neural_network_backward_4/)를 참고하세요.
# 새롭게 정의하는 함수들만 모아놓은 것 입니다. 
# 그외 기존에 사용한 적이 있는 함수는 `source()` 함수를 통해 사용할 수 있습니다.

source("./utils.R")
source("./numerical_gradient.R")
source("./gradient.R")

# 순전파와 비교를 위해 순전파 가중치 계산기능의 함수를 정의합니다.

model.backward.test <- function(x){
  Affine_1_layer <- Affine.forward(W1, b1, x)
  Relu_1_layer <- Relu.forward(Affine_1_layer$out)
  Affine_2_layer <- Affine.forward(W2, b2, Relu_1_layer$out)
  return(list(x  =  Affine_2_layer$out, Affine_1.forward  =  Affine_1_layer, Affine_2.forward  =  Affine_2_layer, Relu_1.forward  =  Relu_1_layer))
}

# 손실함수도 순전파 계산 기능을 이용해 구한 것으로 정의해줍니다.

loss.test <- function(x,t){
  temp  <- model.backward.test(x) 
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(last_layer.forward$loss) 
}


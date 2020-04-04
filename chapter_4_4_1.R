# 이 스크립트는 오차역전파법을 적용한 학습과정을 구현한 것입니다. 
# 스크립트 내 코드의 자세한 설명은 다음 링크(https://choosunsick.github.io/post/neural_network_backward_4/)를 참고하세요.
# 새롭게 정의하는 함수들만 모아놓은 것 입니다. 
# 그외 기존에 사용한 적이 있는 함수는 `source()` 함수를 통해 사용할 수 있습니다.

source("./DeepLearningFromForR/functions.R")
source("./DeepLearningFromForR/utils.R")

# 앞서 2층신경망의 초기값 함수와의 차이점은 초기값을 리스트로 만든다는 점입니다.
# 그외 가중치와 편향이 2개인점과 가중치를 랜덤값으로 설정하는 것은 기존 2층신경망 초기값 함수와 같습니다. 

TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std*matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size),nrow = 1,ncol = hidden_size)
  W2 <- weight_init_std*matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1,ncol = output_size)
  params <<- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  return(list(input_size, hidden_size, output_size,weight_init_std))
}

# 역전파 내에서 사용하는 순전파 계산 모델입니다. 

model.backward <- function(x){
  Affine_1_layer <- Affine.forward(params$W1, params$b1, x)
  Relu_1_layer <- Relu.forward(Affine_1_layer$out)
  Affine_2_layer <- Affine.forward(params$W2, params$b2, Relu_1_layer$out)
  return(list(x  =  Affine_2_layer$out, Affine_1.forward  =  Affine_1_layer, Affine_2.forward  =  Affine_2_layer, Relu_1.forward  =  Relu_1_layer))
}

#손실함수 값을 계산하는 함수입니다.

backward_loss <- function(x, t){
  temp  <- model.backward(x) 
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss  =  last_layer.forward$loss, softmax  =  last_layer.forward, predict  =   temp))
}

# 수치미분이 아닌 방식으로 미분을 진행하는 역전파의 계산 방법입니다. 

gradient <- function(x, t) {
  # 순전파
  temp_loss <- backward_loss(x, t)
  # 역전파
  dout <- 1
  last_layer.backward <- SoftmaxWithLoss.backward(temp_loss$softmax, dout)
  Affine_2_layer.backward <- Affine.backward(temp_loss$predict$Affine_2.forward, dout  =  last_layer.backward$dx)
  Relu_1_layer.backward <- Relu.backward(temp_loss$predict$Relu_1.forward, dout  =  Affine_2_layer.backward$dx)
  Affine_1_layer.backward <- Affine.backward(temp_loss$predict$Affine_1.forward, dout  =  Relu_1_layer.backward$dx)
  
  grads  <- list(W1  =  Affine_1_layer.backward$dW, b1  =  Affine_1_layer.backward$db, W2  =  Affine_2_layer.backward$dW, b2  =  Affine_2_layer.backward$db)
  return(grads)
}
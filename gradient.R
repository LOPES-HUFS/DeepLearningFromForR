# 2층 신경망 구현을 위한 역전파 gradient 함수입니다.
# 이 함수 코드에 대한 설명은 https://choosunsick.github.io/post/neural_network_backward_4/ 에서 찾아보실 수 있습니다.
# 작동에 필요한 함수는 source로 불러와 줍니다.

source("./DeepLearningFromForR/functions.R")

# 역전파 gradient 함수입니다. 

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
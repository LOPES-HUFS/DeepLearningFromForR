# bare backward propagation

## 역전파를 이용하여 네트웍에 MNIST 모델 학습시키기

이 글은 활성화 함수인 `sigmoid()`, `softmax()`, `Relu()`의 순전파, 역전파를 제외한 모든 것을 구현하는 모든 코드를 가지고 있다. 이 코드를 다 이해한다면, 역전파에 대한 기초적인 내용은 알게 되었다고 해도 무방하다. 우선 우리가 사용할 함수들의 코드를 가지고 있는 것을 가져온다.

```R
source("./functions.R")
source("./utils.R")
```

MNIST 자료를 가져오는 방법에 대한 내용은 [Mnist 손글씨 데이터 읽어오는 패키지 소개]을 참고한다. 자료를 가져오는 코드는 아래와 같다. 아래 코드에 대한 소개는 다음을 참고한다.

```R
# install.packages("dslabs") 이미 설치한 것이 있으면 생략
library(dslabs)

mnist_data <- get_data()

x_train_normalize <- mnist_data$x_train
x_test_normalize <- mnist_data$x_test

t_train_onehotlabel <- making_one_hot_label(mnist_data$t_train,60000, 10)
t_test_onehotlabel <- making_one_hot_label(mnist_data$t_test,10000, 10)
```

이제 본격적으로 우리가 학습시킬 네트웍을 만든다.

```R
TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  params <<- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}

TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
```

앞에서 만든 네트웍을 학습시킬 모델을 만든다. 이 함수를 다 따로 만든 이유는 우선 `forward()`은 예측을 하기 위해 필요하다. `loss()`은 당연히 손실값을 알아보기 위해서 필요하다.

```R
forward <- function(x){
  Affine_1_layer <- Affine.forward(params$W1, params$b1, x)
  Relu_1_layer <- Relu.forward(Affine_1_layer$out)
  Affine_2_layer <- Affine.forward(params$W2, params$b2, Relu_1_layer$out)
  return(list(x = Affine_2_layer$out, Affine_1.forward = Affine_1_layer, Affine_2.forward = Affine_2_layer, Relu_1.forward = Relu_1_layer))
}

loss <- function(model.forward, x, t){
  temp <- model.forward(x)
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}


gradient <- function(model.forward, x, t) {
  # 순전파
  temp_loss <- loss(model.forward, x, t)
  # 역전파
  dout <- 1
  last_layer.backward <- SoftmaxWithLoss.backward(temp_loss$softmax, dout)
  Affine_2_layer.backward <- Affine.backward(temp_loss$predict$Affine_2.forward, dout  =  last_layer.backward$dx)
  Relu_1_layer.backward <- Relu.backward(temp_loss$predict$Relu_1.forward, dout  =  Affine_2_layer.backward$dx)
  Affine_1_layer.backward <- Affine.backward(temp_loss$predict$Affine_1.forward, dout  =  Relu_1_layer.backward$dx)
  grads  <- list(W1  =  Affine_1_layer.backward$dW, b1  =  Affine_1_layer.backward$db, W2  =  Affine_2_layer.backward$dW, b2  =  Affine_2_layer.backward$db)
  return(grads)
}
```

지금까지 만든 것을 테스트해보자.

```R
> train_size <- dim(x_train_normalize)[1]
> batch_size <- 100
> grads <- gradient(model.forward=forward, x=x_train_normalize[batch_mask,], t= t_train_onehotlabel[batch_mask,])
> loss_value <- loss(forward=forward, x=x_train_normalize[batch_mask,], t_train_onehotlabel[batch_mask,])$loss
> loss_value
[1] 2.302899
```

실제로 무식하게 돌려봅니다.

```R
for(i in 1:iters_num){
  batch_mask <- sample(train_size ,batch_size)
  x_batch <- x_train_normalize[batch_mask,]
  t_batch <- t_train_onehotlabel[batch_mask,]

  grad <- gradient(model.forward=forward, x_batch, t_batch)
  
  params$W1 <- params$W1 - (grads$W1 * learning_rate)
  params$W2 <- params$W2 - (grads$W2 * learning_rate)
  params$b1 <- params$b1 - (grads$b1 * learning_rate)
  params$b2 <- params$b2 - (grads$b2 * learning_rate)
  }
```

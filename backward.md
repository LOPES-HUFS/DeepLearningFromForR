# 역전파를 이용한 모델 학습 방법

오차역전파를 사용한 학습도 손실함수를 최소화하는 가중치를 찾는 것을 목표로합니다. 다만, 역전파는 가중치를 구함에 있어 연쇄법칙에 기반한 국소적 미분을 활용합니다. 순전파와 비교했을 때 훨씬 빠른 시간 안에 효울적으로 계산한다는 장점이 있습니다. 이번에는 역전파법을 사용하여 모델 학습을 진행해 보겠습니다.

### 학습하기

먼저, 라이브러리와 공통함수를 읽어옵니다.

```R
#install.packages("dslabs")
library(dslabs)

source("./functions.R")
source("./utils.R")
source("./model.R")
source("./layers.R")
```

1개의 은닉층을 갖는 네트워크를 생성합니다. 네트워크는 순전파와 동일합니다.

```R
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
```

데이터를 불러와 트레이닝셋과 테스트셋으로 분리하는 `init()`함수를 생성합니다.
```R
init <- function(){
  mnist_data <- get_data()
  #손글씨 데이터
  x_train_normalize <<- mnist_data$x_train 
  x_test_normalize <<- mnist_data$x_test
  #정답 레이블
  t_train_onehotlabel <<- making_one_hot_label(mnist_data$t_train,60000, 10)
  t_test_onehotlabel <<- making_one_hot_label(mnist_data$t_test,10000, 10)
}
```

앞서 역전파에서는 국소적 미분을 사용한다고 했습니다. 순전파와 반대방향으로 국소적 미분을 곱하여 이전 노드들에 값을 전달하는 것인데, 국소적 미분은 순전파 때의 미분을 구한다는 뜻입니다. 다시 말해, 순전파 때의 미분 값을 구해 다음 노드에 전달하는 함수가 필요합니다.
다음 코드는 순전파 때와 마찬가지로 입력신호와 가중치를 계산하고 Relu함수를 거쳐 다음 노드로 전달합니다.
```R
model.forward <- function(x){
  Affine_1 <- Affine.forward(network$W1, network$b1, x)
  Relu_1 <- Relu.forward(Affine_1$out)
  Affine_2 <- Affine.forward(network$W2, network$b2, Relu_1$out)
  return(list(x = Affine_2$out, Affine_1.forward = Affine_1, Affine_2.forward = Affine_2, Relu_1.forward = Relu_1))
}
```

역전파도 마찬가지로 손실함수를 계산합니다. 
```R
loss <- function(model.forward, x, t){
  temp <- model.forward(x)
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}
```

순전파와 달리 마지막 노드에서부터 거꾸로 계산해 기울기를 구합니다.
```R
gradient <- function(model.forward, x, t) {
  # 순전파
  temp <- loss(model.forward, x, t)
  # 역전파
  dout <- 1
  last.backward <- SoftmaxWithLoss.backward(temp$softmax, dout)
  Affine_2.backward <- Affine.backward(temp$predict$Affine_2.forward, dout  =  last.backward$dx)
  Relu_1.backward <- Relu.backward(temp$predict$Relu_1.forward, dout  =  Affine_2.backward$dx)
  Affine_1.backward <- Affine.backward(temp$predict$Affine_1.forward, dout  =  Relu_1.backward$dx)
  grads  <- list(W1  =  Affine_1.backward$dW, b1  =  Affine_1.backward$db, W2  =  Affine_2.backward$dW, b2  =  Affine_2.backward$db)
  return(grads)
}
```

다음은 학습을 실제로 진행하는 코드입니다.

```R
train_model <- function(batch_size, iters_num, learning_rate, debug=FALSE){
  #seperate train, test data
  init()
  train_size <- dim(x_train_normalize)[1]

  iter_per_epoch <- max(train_size / batch_size)
  for(i in 1:iters_num){
      batch_mask <- sample(train_size ,batch_size)
      x_batch <- x_train_normalize[batch_mask,]
      t_batch <- t_train_onehotlabel[batch_mask,]

      grad <- gradient(model.forward=model.forward, x_batch, t_batch)
      #update weights and biases using SGD
      network <<- sgd.update(network,grad,lr=learning_rate)

      if(debug == TRUE){
          if(i %% iter_per_epoch == 0){
              train_acc <- model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
              test_acc <- model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)
              print(c(train_acc, test_acc))
          }
      }
  }

  train_accuracy = model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
  test_accuracy = model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)
  return(c(train_accuracy, test_accuracy))
}
```

`train_model()`함수 중간에 `sg.update()`함수는 경사하강법으로 변경된 가중치를 업데이트하는 역할을 합니다.
코드는 아래와 같습니다.

```R
sgd.update <- function(network, grads, lr = 0.01){
  for(i in names(network)){network[[i]] <- network[[i]] - (grads[[i]]*lr)}
  return(network)
}
```

이제 모든 준비를 마쳤습니다. 네트워크를 생성한 후 모델을 학습시켜봅니다.
```R
network <<- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
train_model(100, 10000, 0.1, TRUE)
```

위 코드를 실행시키고 3분 정도 지나면 아래와 같은 출력화면이 나올 것입니다. 한 행의 첫 번째 숫자는 훈련데이터 셋에 대한 정확도, 두 번째 숫자는 테스트 셋에 대한 정확도를 나타냅니다. 그리고 하나의 행은 1에폭(epoch)을 의미합니다. 에폭을 진행할수록 정확도가 높아지는 것을 확인할 수 있습니다!

```R
[1] 0.9048 0.9059
[1] 0.9228 0.9247
[1] 0.9355833 0.9343000
[1] 0.9436167 0.9416000
[1] 0.9496167 0.9470000
[1] 0.9563167 0.9519000
[1] 0.9602167 0.9555000
[1] 0.9629167 0.9558000
[1] 0.9664833 0.9603000
[1] 0.9680333 0.9619000
[1] 0.9711167 0.9635000
[1] 0.97315 0.96520
[1] 0.97445 0.96570
[1] 0.9754167 0.9659000
[1] 0.9771167 0.9698000
[1] 0.9779 0.9679
[1] 0.9776833 0.9680000
```

* 에폭에 대한 설명은 [링크](https://choosunsick.github.io/post/neural_network_5/)를 참고하세요.
* 아래의 출력 값들은 초기 값의 랜덤값으로 인해 다른 숫자가 나올 수 있습니다.

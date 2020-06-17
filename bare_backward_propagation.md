# bare backward propagation

## 역전파를 이용하여 네트웍에 MNIST 모델 학습시키기

이 글은 활성화 함수인 `sigmoid()`, `softmax()`, `Relu()`의 순전파, 역전파를 제외한 모든 것을 구현하는 모든 코드를 가지고 있다. 이 코드를 다 이해한다면, 역전파에 대한 기초적인 내용은 알게 되었다고 해도 무방하다. 우선 우리가 사용할 함수들의 코드를 가지고 있는 것을 가져온다.

```R
source("./functions.R")
source("./utils.R")
#source("./optimizer.R")
```

MNIST 자료를 가져오는 방법에 대한 내용은 [Mnist 손글씨 데이터 읽어오는 패키지 소개](https://choosunsick.github.io/post/mnist/)을 참고한다. 자료를 가져오는 코드는 아래와 같다. 아래 코드에 대한 소개는 다음을 참고한다.

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
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std = 0.01) {
  W1 <- weight_init_std*matrix(rnorm(n = input_size*hidden_size), nrow = input_size, ncol = hidden_size)
  b1 <- matrix(rep(0,hidden_size),nrow=1,ncol=hidden_size)
  W2 <- weight_init_std*matrix(rnorm(n = hidden_size*output_size), nrow = hidden_size, ncol = output_size)
  b2 <- matrix(rep(0,output_size),nrow=1,ncol=output_size)
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
```

앞에서 만든 네트웍을 학습시킬 모델을 만든다. 이 함수를 다 따로 만든 이유는 우선 `model.forward()`은 예측을 하기 위해 필요하다. `loss()`은 당연히 손실값을 알아보기 위해서 필요하다.

```R
model.forward <- function(network, x){
  Affine_1 <- Affine.forward(network$W1, network$b1, x)
  Relu_1 <- Relu.forward(Affine_1$out)
  Affine_2 <- Affine.forward(network$W2, network$b2, Relu_1$out)
  return(list(x = Affine_2$out, Affine_1.forward = Affine_1, Affine_2.forward = Affine_2, Relu_1.forward = Relu_1))
}

loss <- function(model.forward, network, x, t){
  temp <- model.forward(network, x)
  y <- temp$x
  last_layer.forward <- SoftmaxWithLoss.forward(y, t)
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}


gradient <- function(model.forward, network, x, t) {
  # 순전파
  temp <- loss(model.forward, network, x, t)
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

지금까지 만든 것을 테스트해보자. 학습 자료 처음부터 100개만 뽑하서 배치를 이용해서 한꺼번에 돌려 손실 함수 값을 계산하겠습니다.


```R
batch_size <- 100

temp_TwoLayerNet <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

grads <- gradient(model.forward=model.forward, network = temp_TwoLayerNet, x=x_train_normalize[1:batch_size,], t= t_train_onehotlabel[1:batch_size,])
loss_value <- loss(model.forward=model.forward, network = temp_TwoLayerNet, x=x_train_normalize[1:batch_size,], t_train_onehotlabel[1:batch_size,])$loss
```

윗 코드를 실행하면 다음과 같은 결과가 나옵니다.

```R
>loss_value
[1] 2.302899
```
이렇게 돌아가는 모델을 평가하는 평가 함수를 만들겠습니다.

```R
model.evaluate <- function(model, network, x, t){
    temp <- model(network, x)
    y <- max.col(temp$x)
    t <- max.col(t)
    accuracy <- (sum(ifelse(y == t,1,0))) / dim(x)[1]
    return(accuracy)
}
```

이 역전파는 SGD를 이용하여 네트웍을 갱신하면서 학습시킬 예정입니다. 이 작업을 할 함수를 만듭니다.

```R
sgd.update <- function(network, grads, lr = 0.01){
  for(i in names(network)){network[[i]] <- network[[i]] - (grads[[i]]*lr)}
  return(network)
}
```

실제로 무식하게 돌려봅니다. 우선 필요한 변수를 설정합니다.

```R
train_size <- dim(x_train_normalize)[1]
batch_mask <- 100
batch_size <- 100
iter_per_epoch <- max(train_size / batch_size)
iters_num <- 10000
```
학습한 자료를 결과를 저장할 장소를 만듭니다.

```R
train_loss_list <- data.frame(lossvalue  =  0)
train_acc_list <- data.frame(train_acc  =  0)
test_acc_list <- data.frame(test_acc  =  0)
```

진짜로 돌립니다. 앞에 iters_num = 10000으로 설정했습니다. 대략 16 애폭정도 돌아갑니다.

```R

epoch_counter <- 0

temp_TwoLayerNet <- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for(i in 1: iters_num){
  batch_mask <- sample(train_size ,batch_size)
  x_batch <- x_train_normalize[batch_mask,]
  t_batch <- t_train_onehotlabel[batch_mask,]
  
  grad <- gradient(model.forward = model.forward, network = temp_TwoLayerNet, x_batch, t_batch)
  
  temp_TwoLayerNet <- sgd.update(temp_TwoLayerNet, grad)
  
  loss_value <- loss(model.forward=model.forward, network = temp_TwoLayerNet, x_batch, t_batch)$loss
  train_loss_list <- rbind(train_loss_list,loss_value)
  
  if(i %% iter_per_epoch == 0){
    epoch_counter <- epoch_counter + 1
    train_acc <- model.evaluate(model.forward, temp_TwoLayerNet, x_train_normalize, t_train_onehotlabel)
    test_acc <- model.evaluate(model.forward, temp_TwoLayerNet, x_test_normalize, t_test_onehotlabel)
    train_acc_list <- rbind(train_acc_list,train_acc)
    test_acc_list <- rbind(test_acc_list,test_acc)
    print(epoch_counter)
    print(c(train_acc, test_acc))
  }
}
```
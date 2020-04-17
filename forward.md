# 순전파

## 학습하기

우선 학습에 필요한 MNIST 데이터를 불러오는데 필요한 패키지를 설치합니다.

```R
install.packages("dslabs")
library(dslabs)
```

학습에 필요한 함수나 코드를 불어옵니다.

```R
source("./functions.R")
source("./utils.R")
source("./numerical_gradient.R")
```

학습할 네트워크를 만듭니다.

```R
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std = 0.01) {
  W1 <<- weight_init_std*matrix(rnorm(n = input_size*hidden_size), nrow = input_size, ncol = hidden_size)
  b1 <<- matrix(rep(0, hidden_size), nrow=1,ncol=hidden_size)
  W2 <<- weight_init_std*matrix(rnorm(n = hidden_size*output_size), nrow = hidden_size, ncol = output_size)
  b2 <<- matrix(rep(0, output_size), nrow=1,ncol=output_size)
  return(list(input_size, hidden_size, output_size, weight_init_std))
}

TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

```R
mnist_data <- get_data()

x_train_normalize <- mnist_data$x_train
x_test_normalize <- mnist_data$x_test

t_train_onehotlabel <- making_one_hot_label(mnist_data$t_train, 60000,10)
t_test_onehotlabel <- making_one_hot_label(mnist_data$t_test, 10000,10)
```
함수를 통해 데이터를 불러옵니다.
학습할 데이터를 처리한 이후 학습에 필요한 파라미터를 설정해줍니다.

```R
learning_rate <- 0.1
iters_num <- 100
train_loss_list <- data.frame(lossvalue=rep(0, iters_num))
train_size <- dim(x_train_normalize)[1]
batch_size <- 100
```

사전 준비는 다 끝났습니다. 실제로 순전파를 이용해서 학습해 보겠습니다.

```R
for(i in 1:iters_num){
  batch_mask <- sample(train_size,batch_size)
  x <- x_train_normalize[batch_mask,]
  t <- t_train_onehotlabel[batch_mask,]
  grads <- numerical_gradient(loss, x, t)
  W1 <- W1 - (grads$W1 * learning_rate)
  W2 <- W2 - (grads$W2 * learning_rate)
  b1 <- b1 - (grads$b1 * learning_rate)
  b2 <- b2 - (grads$b2 * learning_rate)
  loss_value <- loss(x, t)
  train_loss_list[i,1] <- loss_value
}
```

모델을 100번 학습하는데 걸리는 시간은 3시간정도 소요가 됩니다. 참고로 파이썬에서 순전파 학습을 100번 하는데에는 약 5300초가 걸립니다. 결과는 다음과 같습니다.

## 모델평가

이제 손실 함수 값과 모델의 정확도를 확인해 봅시다.
```R
train_loss_list

model.evaluate(x_test_normalize,t_test_onehotlabel)
```

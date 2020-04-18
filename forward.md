# 순전파를 이용한 모델 학습 방법

순전파는 원리만 이해하면 됩니다. 굳이 순전파를 사용하여 데이터를 학습하는 것은 어쩌면 무모한 짓입니다. 역전파법을 사용하면 3분이면 될 것을 8시간에 걸쳐서 학습하기 때문입니다.😂

그럼에도 불구하고 구현해 보았습니다. 순전파로 학습하기...!

마찬가지로 손글씨로 쓴 이미지를 판별하는 딥러닝 구현 과정을 살펴보면서, 순전파의 원리와 함께 순전파가 느린 이유를 풀어 설명해 보도록 하겠습니다.


## 학습하기

우선 학습에 필요한 MNIST 데이터를 불러오는데 필요한 패키지를 설치합니다.

```R
install.packages("dslabs")
library(dslabs)
```

학습에 필요한 함수나 코드를 불어옵니다. 아래 소스코드들을 불러오기 위해 경로 설정을 해주어야 합니다. 경로 설정에 대한 부분은 [링크](https://github.com/LOPES-HUFS/DeepLearningFromForR/blob/master/README.md)의 프로젝트 맛보기를 참고해주세요.

```R
source("./functions.R")
source("./utils.R")
source("./numerical_gradient.R")
source("./TwoLayerNet_model.forward.R")
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
```

각 파라미터의 의미는 아래와 같습니다.
 * input_size : 입력 노드의 개수로, 여기서는 한 이미지의 크기(28*28)를 의미합니다. 
 * hidden_size : 은닉층의 노드 개수로, 여기서는 50개로 설정하였습니다.
 * output_size : 출력 노드의 개수로, 숫자 0~9의 값을 분류하기 때문에 10이 됩니다.
 * weight_init_std : 가중치 초기값이 큰 값이 되는 것을 방지하는 파라미터입니다.


이제, 손글씨 이미지를 불러오고 학습을 위해 훈련 셋과 테스트 셋으로 나눕니다.
```R
mnist_data <- get_data()

x_train_normalize <- mnist_data$x_train
x_test_normalize <- mnist_data$x_test

t_train_onehotlabel <- making_one_hot_label(mnist_data$t_train, 60000,10)
t_test_onehotlabel <- making_one_hot_label(mnist_data$t_test, 10000,10)
```

학습에 필요한 파라미터를 설정합니다.

```R
learning_rate <- 0.1
iters_num <- 100
train_loss_list <- data.frame(lossvalue=rep(0, iters_num))
train_size <- dim(x_train_normalize)[1]
batch_size <- 100
```
 각 변수의 의미는 다음과 같습니다.

 * learning_rate : 학습률로, 학습률이 높을수록 학습이 빨리 진행되는 대신에 덜 진행될 수 있습니다.
 * iters_num : 학습 반복 횟수
 * train_loss_list : 손실 함수 값 기록 리스트
 * train_size : 전체 훈련 셋 개수
 * batch_size : 훈련 셋에서 뽑을 이미지 개수

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

모델을 100번 학습하는데 걸리는 시간은 최소 3시간 이상이 소요가 됩니다. 참고로 [파이썬 코드](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch04/two_layer_net.py)를 반복횟수 100번으로 설정하여 실행시키면 약 5300초가 걸립니다.

## 모델평가

이제 손실 함수 값과 모델의 정확도를 확인해 봅시다.

```R
train_loss_list

model.evaluate.forward(x_test_normalize,t_test_onehotlabel)
```

# 역전파를 이용한 모델 학습 방법

```R
install.packages("dslabs")
library(dslabs)
```

여기까지 진행하시면 기초 설정은 다 끝났습니다. 아래의 코드를 직접 입력해 보며 결과를 확인할 수 있을 것입니다. 이해는 잠시 뒤로한 채 가볍게 읽어 주세요.

### 학습하기

이제 모델에 손글씨 데이터를 학습시켜 봅시다. 여기서는 역전파를 이용하여 학습 해보겠습니다.

```R
source("./functions.R")
source("./utils.R")
source("./gradient.R")
source("./TwoLayerNet_model.backward.R")

TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  params <<- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  return(list(input_size, hidden_size, output_size, weight_init_std))
}

TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
```

우선 필요한 함수를 불러오고 모델의 초기값을 생성해줍니다. 다음에는 학습할 데이터를 가져옵니다.

```R
mnist_data <- get_data()

x_train_normalize <- mnist_data$x_train
x_test_normalize <- mnist_data$x_test

t_train_onehotlabel <- making_one_hot_label(mnist_data$t_train,60000, 10)
t_test_onehotlabel <- making_one_hot_label(mnist_data$t_test,10000, 10)
```

학습할 데이터를 처리해주고 나면 학습을 위한 파라미터를 설정해줍니다.

```R
iters_num <- 10000
train_size <- dim(x_train_normalize)[1]
batch_size <- 100
learning_rate <- 0.1

train_loss_list <- data.frame(lossvalue  =  rep(0,iters_num))
train_acc_list <- data.frame(train_acc  =  0)
test_acc_list <- data.frame(test_acc  =  0)

iter_per_epoch <- max(train_size / batch_size)
```

다음은 학습을 실제로 진행하는 코드입니다.

```R
for(i in 1:iters_num){
  batch_mask <- sample(train_size ,batch_size)
  x_batch <- x_train_normalize[batch_mask,]
  t_batch <- t_train_onehotlabel[batch_mask,]

  grad <- gradient(x_batch, t_batch)

  params$W1 <- params$W1 - (grad$W1 * learning_rate)
  params$W2 <- params$W2 - (grad$W2 * learning_rate)
  params$b1 <- params$b1 - (grad$b1 * learning_rate)
  params$b2 <- params$b2 - (grad$b2 * learning_rate)

  loss_value <- loss(x_batch, t_batch)$loss
  train_loss_list <- rbind(train_loss_list, loss_value)

  if(i %% iter_per_epoch == 0){
    train_acc <- model.evaluate.backward(x_train_normalize, t_train_onehotlabel)
    test_acc <- model.evaluate.backward(x_test_normalize, t_test_onehotlabel)
    train_acc_list <- rbind(train_acc_list, train_acc)
    test_acc_list <- rbind(test_acc_list, test_acc)
    print(c(train_acc, test_acc))
  }
}
```

위 코드를 실행시키고 3분 정도 지나면 아래와 같은 출력화면이 나올 것입니다. 한 행의 첫 번째 숫자는 훈련데이터 셋에 대한 정확도, 두 번째 숫자는 테스트 셋에 대한 정확도를 나타냅니다. 그리고 하나의 행은 1에폭(epoch)을 의미합니다. 에폭을 진행할수록 정확도가 높아지는 것을 확인할 수 있습니다.

* 에폭에 대한 설명은 [링크](https://choosunsick.github.io/post/neural_network_5/)를 참고하세요.
* 아래의 출력 값들은 초기 값의 랜덤값으로 인해 다른 숫자가 나올 수 있습니다.

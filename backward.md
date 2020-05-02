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

# 네트워크 만들기
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
```

우선 필요한 함수를 불러오고 모델의 초기값을 생성해줍니다. 다음에는 학습할 데이터를 가져옵니다.

```R
init <- function(){
  mnist_data <- get_data()
  x_train_normalize <<- mnist_data$x_train 
  x_test_normalize <<- mnist_data$x_test
  t_train_onehotlabel <<- making_one_hot_label(mnist_data$t_train,60000, 10)
  t_test_onehotlabel <<- making_one_hot_label(mnist_data$t_test,10000, 10)
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

      grad <- gradient(x_batch, t_batch)
      #update weights and biases using SGD
      network <<- sgd.update(network,grad,lr=learning_rate)

      if(debug == TRUE){
          if(i %% iter_per_epoch == 0){
              train_acc <- model.evaluate.backward(x_train_normalize, t_train_onehotlabel)
              test_acc <- model.evaluate.backward(x_test_normalize, t_test_onehotlabel)
              print(c(train_acc, test_acc))
          }
      }
  }

  train_accuracy = model.evaluate.backward(x_train_normalize, t_train_onehotlabel)
  test_accuracy = model.evaluate.backward(x_test_normalize, t_test_onehotlabel)
  return(c(train_accuracy, test_accuracy))
}

network <<- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
train_model(100, 10000, 0.1, TRUE)
```

위 코드를 실행시키고 3분 정도 지나면 아래와 같은 출력화면이 나올 것입니다. 한 행의 첫 번째 숫자는 훈련데이터 셋에 대한 정확도, 두 번째 숫자는 테스트 셋에 대한 정확도를 나타냅니다. 그리고 하나의 행은 1에폭(epoch)을 의미합니다. 에폭을 진행할수록 정확도가 높아지는 것을 확인할 수 있습니다.

* 에폭에 대한 설명은 [링크](https://choosunsick.github.io/post/neural_network_5/)를 참고하세요.
* 아래의 출력 값들은 초기 값의 랜덤값으로 인해 다른 숫자가 나올 수 있습니다.

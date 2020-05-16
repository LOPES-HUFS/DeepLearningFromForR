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

학습에 사용되는 공통 함수를 불러옵니다. 아래 소스코드들을 불러오기 위해 경로 설정을 해주어야 합니다. 경로 설정에 대한 부분은 [링크](https://github.com/LOPES-HUFS/DeepLearningFromForR/blob/master/README.md)의 프로젝트 맛보기를 참고해주세요.

```R
source("./functions.R")
source("./utils.R")
source("./model.R")
```

먼저, 학습할 네트워크를 만듭니다. W1,W2는 각 층별 가중치이며 b1,b2는 편향 값을 의미합니다.
```R
TwoLayerNet <- function(input_size, hidden_size, output_size, weight_init_std  =  0.01) {
  W1 <- weight_init_std * matrix(rnorm(n  =  input_size*hidden_size), nrow  =  input_size, ncol  =  hidden_size)
  b1 <- matrix(rep(0,hidden_size), nrow = 1, ncol = hidden_size)
  W2 <- weight_init_std * matrix(rnorm(n  =  hidden_size*output_size), nrow  =  hidden_size, ncol  =  output_size)
  b2 <- matrix(rep(0,output_size),nrow = 1, ncol = output_size)
  
  return (list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
```

각 파라미터의 의미는 아래와 같습니다.
 * input_size : 입력 노드의 개수로, 여기서는 한 이미지의 크기(28*28)를 의미합니다. 
 * hidden_size : 은닉층의 노드 개수로, 여기서는 50개로 설정하였습니다.
 * output_size : 출력 노드의 개수로, 숫자 0~9의 값을 분류하기 때문에 10이 됩니다.
 * weight_init_std : 가중치 초기값이 큰 값이 되는 것을 방지하는 파라미터입니다.

다음으로, 데이터를 불러오고 트레이닝 셋과 테스트 셋으로 분류합니다. 데이터는 MNIST 라이브러리의 손글씨 이미지입니다. 
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

학습에 필요한 파라미터를 설정합니다.

손실함수는 교차엔트로피오차 함수를 사용합니다. 교차엔트로피오차 함수는 아래와 같이 구현합니다.
```R
model.forward <- function(x){
  z1 <- sigmoid(sweep((x %*% network$W1),2, network$b1,'+'))
  return(softmax(sweep((z1 %*% network$W2),2, network$b2,'+')))
}

cross_entropy_error <- function(y, t){
    delta <- 1e-7
    batchsize <- dim(y)[1]
    return(-sum(t * log(y + delta))/batchsize)
}

loss <-function(x,t){
  return(cross_entropy_error(model.forward(x),t))
}
```
기본 교차엔트로피 함수식에 `delta`값을 추가하였는데, 이는 log0이 되면 -Inf가 되는 문제를 방지하기 위해서 입니다.

다음으로 경사하강법은 손실함수 값을 최소화 시키기 위해 사용합니다. 
```R
numerical_gradient_W <- function(f,x,t,weight){
    h <- 1e-4
    vec <- matrix(0, nrow = nrow(network[[weight]]) ,ncol = ncol(network[[weight]]))
    for(i in 1:length(network[[weight]])){
        origin <-  network[[weight]][i]
        network[[weight]][i] <<- (network[[weight]][i] + h)
        fxh1 <- f(x, t)
        network[[weight]][i] <<- (network[[weight]][i] - (2*h))
        fxh2 <- f(x, t)
        vec[i] <- (fxh1 - fxh2) / (2*h)
        network[[weight]][i] <<- origin
    }
    return(vec)
}

numerical_gradient <- function(f,x,t) {
  grads  <- list(W1 = numerical_gradient_W(f,x,t,"W1"), 
                 b1 = numerical_gradient_W(f,x,t,"b1"), 
                 W2 = numerical_gradient_W(f,x,t,"W2"), 
                 b2 = numerical_gradient_W(f,x,t,"b2"))
  return(grads)
}
```

마지막으로 학습시키는 함수입니다.
```R
train_model <- function(batch_size, iters_num, learning_rate, debug=FALSE){
  #seperate train, test data
  init()
  train_size <- dim(x_train_normalize)[1]

  iter_per_epoch <- max(train_size / batch_size)
  for(i in 1:iters_num){
    batch_mask <- sample(train_size,batch_size)
    x_batch <- x_train_normalize[batch_mask,]
    t_batch <- t_train_onehotlabel[batch_mask,]

    grad <- numerical_gradient(loss, x_batch, t_batch)
    network <<- sgd.update(network,grad,lr=learning_rate)

    if(debug){
        if(i %% iter_per_epoch == 0){
            train_acc <- model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
            test_acc <- model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)
            print(c(train_acc, test_acc))
        }
    }

    train_accuracy = model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
    test_accuracy = model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)
    return(c(train_accuracy, test_accuracy))
    }
}
```

`train_model()`함수 중간에 `sg.update()`함수는 경사하강법으로 변경된 가중치를 업데이트하는 역할을 합니다.
코드는 아래와 같습니다.
```R
sgd.update <- function(network, grads, lr = 0.01){
  for(i in names(network)){
      print(i)
      network[[i]] <- network[[i]] - (grads[[i]]*lr)
      }
  return(network)
}
```

이제 모든 준비를 마쳤습니다. 네트워크를 생성한 후 모델을 학습시켜봅니다.
```R
network <<- TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
train_model(100, 10000, 0.1, TRUE)
```

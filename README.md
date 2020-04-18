# 손으로 쓴 숫자 이미지를 판별하는 딥러닝을 R로만 구현하기

## 프로젝트의 목표

이 프로젝트는 딥러닝 분야에서 유명한 [MNIST 데이터베이스](http://yann.lecun.com/exdb/mnist/)를 이용하여 손글씨로 쓴 숫자 이미지를 어떤 숫자인지 추론할 수 있는 Deep Learning(이하 딥러닝)을 R로만 구현하는 것을 목표로 진행합니다. 전반적인 프로젝트 내용은 한빛미디어에서 출판한 [『밑바닥부터 시작하는 딥러닝』](http://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)과 이 책의 최근 Python(이하 파이썬) 코드인 [『ゼロから作る Deep Learning』(O'Reilly Japan, 2016)](https://github.com/oreilly-japan/deep-learning-from-scratch)를 참고하였습니다. 참고 서적의 파이썬 코드를 R로 변환하여 ML을 구현하고 있으며, R에 없는 파이썬의 일부 기능은 R에 맞춰서 변경하였습니다.

앞서 언급했듯이 전체 코드는 [R](https://www.r-project.org)로 구현했습니다. 딥러닝을 구현하려면 행렬을 계산하는 프로그래밍이 필요한데, 이런 개념을 가장 쉽게 구현할 수 있는 언어가 R입니다. R이라는 언어를 모르시는 분들을 위해서 기초 가이드를 만들었습니다. 아래 [R 기초 가이드](https://github.com/LOPES-HUFS/DeepLearningFromForR#r-초심자를-위한-r-설치방법과-기초-사용방법)를 참고하세요. [R 설치하기](https://choosunsick.github.io/post/r_install/)부터 차례대로 따라와 주시면 됩니다. :smile: 한 번씩 읽으시면 이 프로젝트를 이해하는데 많은 도움이 됩니다. 만약 R이 처음이 아니라면 아래링크를 쭉 보면서 따라오시면 기본적인 지식을 습득하실 수 있습니다.

[LOPES](http://www.lopes.co.kr)팀에서 이 프로젝트를 하게 된 이유는 딥러닝 기초 서적을 공부하던 도중, 딥러닝을 R로 직접 구현해보면 R 프로그래밍 실력도 늘릴 수 있고, 딥러닝 기초 원리도 좀 더 명확하게 이해할 수 있을 것 같아서 출발하게 되었습니다. 교육 목적으로 시작한 프로젝트이기 때문에 R로 딥러닝를 구현하는 강좌도 따로 쓰고 있습니다. 아래 [손으로 쓴 숫자 이미지를 판별하는 딥러닝을 R로만 구현하는 방법 소개](https://github.com/LOPES-HUFS/DeepLearningFromForR#손으로-쓴-숫자-이미지를-판별하는-딥러닝을-r로만-구현하는-방법-소개)에 있는 글을 읽어보시면 우리 팀이 R로 어떻게 딥러닝을 구현하고 있는지 살펴보실 수 있습니다. 또한 R을 어느 정도 알고 있지만, 딥러닝에 대해서 잘 알지 못하신다면, 딥러닝 기초 원리를 파악하시는데 도움이 될 것 같습니다.

## 프로젝트 맛보기

어떤 내용을 학습하게 될지 궁금하신 분들을 위해 프로젝트 맛보기를 준비했습니다. 프로젝트 전체 코드를 다운 받은 후, 아래 코드를 따라 입력해 보시는 걸 추천드립니다. 신기하거든요! :smile: 프로젝트 전체 코드를 다운 받으시려면, 이 프로젝트의 [메인 페이지](https://github.com/LOPES-HUFS/DeepLearningFromForR) 오른쪽 상단에 'Clone or download' 버튼을 클릭 후 'Download ZIP' 버튼을 눌러 다운 받으시면 됩니다. 만약, [git](https://git-scm.com/downloads)을 설치하셨으면 아래의 명령어를 커맨드 창에 입력해도 다운 받을 수 있습니다.

```bash
git clone https://github.com/LOPES-HUFS/DeepLearningFromForR.git
```
그리고 파일이 설치된 경로는 아래의 명령어로 확인할 수 있습니다.
```bash
git clone https://github.com/LOPES-HUFS/DeepLearningFromForR.git
```

코드를 다운 받으셨다면, R 에디터에서 디렉토리 설정을 진행해야 합니다. R를 연 후에 다음과 같이 디렉토리 경로를 입력해주세요. 
```R
setwd('<PATH>/DeepLearningFromForR')
```
`<PATH>` 대신에 DeepLearningFromForR 폴더가 위치한 경로를 넣으면 되는데 경로를 모르신다면 다음과 같이 해보세요.
먼저, 커맨드 창에 아래의 명령어를 쳐서 현재 경로를 확인합니다.
```bash
pwd //현재 경로 확인
ls -al // 현재 경로에 있는 폴더 및 파일 확인
```
현재 경로에서 폴더가 있는 경로까지 이동해 주시면 되는데, 이동하는 명령어는 `cd` 입니다.

다음으로 이 프로젝트에서 사용할 MNIST 데이터를 R 패키지 기능을 사용해서 아래 코드를 입력하여 설치합니다. 설치에 대한 자세한 내용은 [Mnist 손글씨 데이터 읽어오는 패키지 소개](https://choosunsick.github.io/post/mnist/)를 참조하시면 좋습니다.

```R
install.packages("dslabs")
library(dslabs)
```

여기까지 진행하시면 기초 설정은 다 끝났습니다. 아래의 코드를 직접 입력해 보며 결과를 확인할 수 있을 것입니다. 이해는 잠시 뒤로한 채 가볍게 읽어 주세요.

### 데이터 학습하기

이제 모델에 손글씨 데이터를 학습시켜 봅시다. 여기서는 역전파를 이용하여 학습 해보겠습니다. 자세한 내용에 대해서는 [오차역전파법을 적용하기](https://choosunsick.github.io/post/neural_network_backward_4/)를 참조해주시기 바랍니다. 다른 학습 방법은 [다양한 학습 방법](https://github.com/LOPES-HUFS/DeepLearningFromForR#다양한-학습-방법)을 참고하세요.

```R
source("./functions.R")
source("./utils.R")
source("./gradient.R")

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
  
  loss_value <- backward_loss(x_batch, t_batch)$loss
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

위 코드를 실행시키고 3분 정도 지나면 아래와 같은 출력화면이 나올 것입니다. 참고로 행의 첫 번째 숫자는 훈련데이터 셋의 예측값이고 행의 두 번째에 나오는 숫자는 테스트 셋에 대한 예측 값입니다. 하나의 행은 1에폭(epoch)을 의미합니다. 에폭이 무엇인지 궁금하신 분은 [링크](https://choosunsick.github.io/post/neural_network_5/)를 참조하세요. 아래의 출력 값들은 초기 값의 랜덤값으로 인해 다른 숫자가 나올 수 있습니다.

```R
[1] 0.9012333 0.9055000
[1] 0.9225333 0.9236000
[1] 0.9380167 0.9360000
[1] 0.9459667 0.9428000
[1] 0.9528667 0.9496000
[1] 0.9568 0.9525
[1] 0.9602833 0.9560000
[1] 0.9642333 0.9592000
[1] 0.9674333 0.9600000
[1] 0.9692167 0.9633000
[1] 0.97025 0.96430
[1] 0.9729833 0.9653000
[1] 0.9740167 0.9653000
[1] 0.9757333 0.9660000
[1] 0.9769 0.9677
[1] 0.9782833 0.9691000
```

약 97%의 성능의 모델을 얻을 수 있습니다. 이제 이 모델을 가지고 숫자를 예측해봅시다. 위 학습과정의 자세한 설명은 아래 [손으로 쓴 숫자 이미지를 판별하는 딥러닝을 R로만 구현하는 방법 소개](https://github.com/LOPES-HUFS/DeepLearningFromForR#손으로-쓴-숫자-이미지를-판별하는-딥러닝을-r로만-구현하는-방법-소개)에서 **4.오차역전파법** 항목을 확인해주세요.

### 숫자 맞추기

위 모델이 MNIST의 숫자 이미지를 사용하여 어떻게 숫자를 예측하는지 살펴 보겠습니다. 먼저 MNIST 숫자 이미지를 확인합니다.

```R
draw_image(mnist_data$x_train[2,])
```

![숫자0](https://user-images.githubusercontent.com/19144813/79545694-13144280-80cc-11ea-8a9a-69c71ad298ed.png)

위 숫자 이미지는 0입니다. R은 1부터 표시하기 때문에 정답 레이블은 1이됩니다. 과연 이 모델은 숫자 0을 맞출 수 있을까요?

```R
> predict.backward(x_train_normalize[2,])
          [,1]        [,2]        [,3]         [,4]         [,5]         [,6]         [,7]         [,8]         [,9]
[1,] 0.9982558 1.44705e-10 0.001739177 2.583026e-06 1.293144e-11 9.714383e-07 1.949852e-07 3.216493e-07 5.454262e-07
            [,10]
[1,] 3.788493e-07
```
보시다 싶이 첫번째 인덱스를 99%확률로 표기합니다. 즉 숫자 0으 맞춘 것입니다. 
### 전체 테스트 셋 추론하기

이제 위 모델이 전체 테스트셋 이미지 1만장을 얼마나 잘 맞추는지 확인해 보겠습니다.

```R
model.evaluate.backward(x_test_normalize,t_test_onehotlabel)
[1] 0.9698
```

여기서 결과값은 0.9698로 인공지능이 숫자를 맞출 확률이 약 97%임을 의미합니다. 학습을 더 많이 반복하거나 합성곱과 같은 방법을 사용한다면 정확독 무려 99%까지도 가능합니다! 차후 프로젝트를 진행해 나가면서 정확도를 99%까지 올리는 것도 같이 확인해 보겠습니다.

## 다양한 학습 방법

네트웍을 학습하는 방법은 다음과 같습니다.

1. [순전파를 이용한 학습 방법](https://github.com/LOPES-HUFS/DeepLearningFromForR/blob/master/forward.md)

## 손으로 쓴 숫자 이미지를 판별하는 딥러닝을 R로만 구현하는 방법 소개

앞에서도 언급했지만, 이 프로젝트는 이 프로젝트를 같이 진행 중인 [choosunsick](https://github.com/choosunsick)이 지금까지 진행된 프로젝트 내용을 정리하고 글로 쓰고 있습니다. 아래 링크를 쭉 살펴보시면, 어떻게 구현하고 있는지 아실 수 있습니다.

1. 벡터와 행렬의 연산
   1. [벡터 연산](https://choosunsick.github.io/post/vector_operation/)
   2. [행렬 연산](https://choosunsick.github.io/post/matrix_operation/)
   3. [브로드 캐스트](https://choosunsick.github.io/post/broadcast_operation/)
2. 신경망
   1. [신경망 소개](https://choosunsick.github.io/post/neural_network_intro/)
   2. [활성화 함수 소개](https://choosunsick.github.io/post/activation_fuctions/)
   3. [3층 신경망 구현](https://choosunsick.github.io/post/softmax_function/)
   4. [신경망 연습 - 손글씨 인식하기](https://choosunsick.github.io/post/neural_network_practice/)  
3. 신경망 학습과정
   1. [신경망 학습이론](https://choosunsick.github.io/post/neural_network_1/)
   2. [손실함수](https://choosunsick.github.io/post/neural_network_2/)
   3. [미니배치 학습](https://choosunsick.github.io/post/neural_network_3/)
   4. [미분과 경사하강법](https://choosunsick.github.io/post/neural_network_4/)
   5. [학습알고리즘 구현](https://choosunsick.github.io/post/neural_network_5/)
4. 오차역전파법
   1. [계산그래프와 연쇄법칙](https://choosunsick.github.io/post/neural_network_backward_1/)
   2. [역전파 예제](https://choosunsick.github.io/post/neural_network_backward_2/)
   3. [다양한 역전파 계층](https://choosunsick.github.io/post/neural_network_backward_3/)
   4. [역전파 적용하기](https://choosunsick.github.io/post/neural_network_backward_4/)

## R 초심자를 위한 R 설치방법과 기초 사용방법

R이 처음이신 분들을 위한 가이드를 작성해 보았습니다. R 설치하기부터 차례대로 따라와 주시면 됩니다:)

1. [R 설치하기](https://choosunsick.github.io/post/r_install/)
2. [자료형 살펴보기](https://choosunsick.github.io/post/r_structure/)
3. [대표 문법 살펴보기](https://choosunsick.github.io/post/r_programming_grammar/)

## 나가면서

프로젝트를 함께 따라오다 보면 어느새 작은 인공지능을 만들고 그 원리를 이해할 수 있을 것입니다.
딥러닝을 시작하기 전에는 수학부터 해야 돼, 코딩부터 해야돼 라면서 딥러닝 공부를 미루지 마세요. 딥러닝 원리를 이해하면서 수학도 코딩도 같이 이해해 나가면 됩니다.

같이 공부해요! :smile:

* 오타 발견 및 문의 사항은 언제나 환영하며 Issues에 등록 부탁드립니다.
  * Issues : [https://github.com/LOPES-HUFS/DeepLearningFromForR/issues]([https://github.com/LOPES-HUFS/DeepLearningFromForR/issues])
* DeepLearningFromForR 프로젝트는 [한국외국어대학교 LOPES 스터디 그룹](http://lopes.hufs.ac.kr)에서 진행합니다.

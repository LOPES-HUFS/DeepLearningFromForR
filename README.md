## Introduction

이 프로젝트는 R언어를 사용하여 딥러닝 기초부터 차근차근 공부할 수 있게 도움을 주고자 출발한 프로젝트입니다.
딥러닝이 처음이신 분, 코딩이나 수학이 어려우신 분들을 대상으로 진행됩니다.

프로젝트는 기계 학습 분야에서 가장 기본적이고 유명한 [MNIST 데이터베이스](http://yann.lecun.com/exdb/mnist/)를 이용하여  숫자 이미지를 어떤 숫자인지 추론할 수 있는 인공지능을 만듭니다.
손글씨 판별 인공지능을 직접 구현해보며 컴퓨터가 이미지를 어떻게 인식하는지, 그리고 손글씨 이미지를 어떻게 맞출 수 있는지 그 과정을 살펴봅니다.

전체 코드는 R언어로 구현했습니다. 딥러닝을 구현하려면 행렬을 계산할 프로그래밍이 필요한데, 이런 개념을 가장 쉽게 구현할 수 있는 언어가 R입니다. R이라는 언어를 모르시는 분들을 위해서 기초 가이드를 만들었습니다. 아래 [R 기초 가이드](https://github.com/LOPES-HUFS/DeepLearningFromForR#r-기초-가이드)를 참고하세요. 한번씩 읽으시면 이 프로젝트를 이해하는데 많은 도움이 됩니다. 만약 R이 처음이 아니라면 아래 링크를 쭉 보면서 따라오시면 기본적인 지식을 습득하실 수 있어요.

1. R 시작하기
2. 벡터와 행렬 살펴보기
3. ...

### 프로젝트 맛보기!
앞서 설명드린 MNIST의 숫자 이미지를 사용하여 어떻게 숫자를 예측하는지 간략하게 살펴 보겠습니다. 코드는 챕터를 진행하면서 풀어 설명드리겠습니다. 먼저 MNIST 숫자 이미지를 확인합니다. 

```R
library(dslabs)
mnist <- read_mnist()
x_train <- mnist$train$images

draw_image <- function(x){
    return(image(1:28, 1:28, matrix(x, nrow=28)[ , 28:1], col = gray(seq(0, 1, 0.05)), xlab="", ylab=""))
    }

draw_image(x_train[2,])
```

<img src="./images/chapter5_mnist_image.png" width="200px">

위 숫자 이미지는 7입니다. MNIST에는 이런 이미지가 7만개가 있습니다. 우리는 이 이미지를 보고 7이라는 것을 바로 알지만, 인공지능은 학습이 필요합니다. 인공지능을 학습할 때에는 이미지를 행렬로 변환하여 학습시킵니다. 변환된 행렬은 다음과 같습니다.

```R
print(x_train[1,])
```
<img src="./images/chapter5_image_to_matrix.png" width="400px">

이미지를 행렬로 변환시킨 후에 인공지능이 숫자를 예측하게 하고 맞았는지,틀렸는지 알려주면서 학습 시킵니다. 학습 후에는 얼마나 잘 맞추는지 확인합니다.

```R
source('./sample/chapter5_sample.R')

x_test <- mnist$test$images
t_test <- mnist$test$labels

x_test_normalize <- x_test/255
t_test_onehotlabel <- making_one_hot_label(t_test,10000,10)

batch_mask <- sample(10000,100)

x <- x_test_normalize[batch_mask,]
t <- t_test_onehotlabel[batch_mask,]

model.evaluate(x, t)
```

<img src="./images/chapter5_accuracy_result.png" width="200px">

여기서 결과값은 0.92로 인공지능이 숫자를 맞출 확률이 92%임을 의미합니다. 학습을 더 많이 반복하거나 학습법을 개선한다면 무려 99%까지도 가능합니다! 

프로젝트를 함께 따라오다보면 어느새 작은 인공지능을 만들고 그 원리를 이해할 수 있을 것입니다. 
딥러닝을 시작하기 전에는 수학부터 해야 돼, 코딩부터 해야돼 라면서 딥러닝 공부를 미루지 마세요. 딥러닝 원리를 이해하면서 수학도 코딩도 같이 이해해 나가면 됩니다.

같이 공부해요!:)

* 프로젝트에서 소개되는 코드는 [밑바닥부터 시작하는 딥러닝](http://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)과 [이 책의 github](https://github.com/WegraLee/deep-learning-from-scratch)를 참고하여 R로 작성하였으며, 일부는 R에 맞게 일부 각색하였습니다.
* 오타 발견 및 문의 사항은 언제나 환영하며 Issues에 등록 부탁드립니다.

    Issues : https://github.com/LOPES-HUFS/DeepLearningFromForR/issues
* DeepLearningFromForR 프로젝트는 한국외국어대학교 LOPES 스터디 그룹에서 진행합니다.

## R 기초 가이드

1. [R 초심자를 위한 R 설치방법과 기초 사용방법](https://choosunsick.github.io/post/r_install/)
2. ...

#구현한 3층신경망을 적용하여 손글씨 인식하는 모델을 만드는 스크립트입니다. 
# 코드 설명과 사용은 다음 링크(https://choosunsick.github.io/post/neural_network_practice/)를 참조하세요.
# 
# 기존의 sigmoid 함수와 softmax 등의 함수를 불러옵니다.
# 3층 신경망 모델의 계산 방법을 불러옵니다. 

source("./DeepLearningFromForR/functions.R")
source("./DeepLearningFromForR/chapter_2_3.R")

# 모델의 성능을 정확도로 평가하는 함수입니다.   
# single 함수의 경우 이미지를 1장씩 비교해 정확도를 계산합니다. 

model.evaluate.single <- function(model,x,t){
  y <- do.call(rbind,lapply(1:NROW(x),function(i)max.col(model.forward(model,x[i,]))))
  t <- max.col(t)
  accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
  return(accuracy)
}

# 한번에 여러장의 이미지를 비교해 정확도를 계산합니다. 

model.evaluate <- function(model,x,t){
  y <- max.col(model.forward(model,x))
  t <- max.col(t)
  accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
  return(accuracy)
}
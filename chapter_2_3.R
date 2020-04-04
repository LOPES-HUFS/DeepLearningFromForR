#3층신경망 구현 스크립트입니다. 
#코드 설명과 사용은 다음 링크(https://choosunsick.github.io/post/softmax_function/)를 참조하세요.

#3층 신경망 모델의 초깃값을 만들어 주는 함수입니다. 
#이 초기값은 3층인 신경망 모델에서만 사용됩니다. 
init <- function(){
  W1 <- matrix(seq(0.1,0.6,0.1), nrow = 2, ncol = 3)
  b1 <- matrix(seq(0.1,0.3,0.1), nrow = 1, ncol = 3)
  W2 <- matrix(seq(0.1,0.6,0.1), nrow = 3, ncol = 2)
  b2 <- matrix(c(0.1, 0.2), nrow = 1, ncol = 2)
  W3 <- matrix(seq(0.1,0.4,0.1), nrow = 2, ncol = 2)
  b3 <- matrix(c(0.1, 0.2), nrow = 1,ncol = 2)
  model <- list(W1, b1, W2, b2, W3, b3)
  names(model) <- c("W1", "b1", "W2", "b2", "W3", "b3")
  return(model)
}

#3층 신경망 모델 계산 과정입니다.
#초기값을 생성한 모델과 입력 데이터 값을 인자로 받아 사용합니다.  
model.forward <- function(model, x){
  a1 <- sweep(x %*% model$W1,2,model$b1,"+")
  z1 <- sigmoid(a1)
  a2 <- sweep(z1 %*% model$W2,2,model$b2,"+")
  z2 <- sigmoid(a2)
  a3 <- sweep(z2 %*% model$W3,2,model$b3,"+")
  return(identify_fun(a3))
}

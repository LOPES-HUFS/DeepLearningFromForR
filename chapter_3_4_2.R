# 이 스크립트는 2층 신경망의 학습과정을 구현한 것입니다. 
# 스크립트 내 코드와 관련된 설명은 다음 링크(https://choosunsick.github.io/post/neural_network_5/)에서 확인할 수 있습니다.
# 이 챕터에서 새롭게 정의한 함수만 스크립트에 저장됩니다. 
# 그외 사용한 적이 있는 함수는 source() 코드를 통해 불러와 줍니다.

source("./DeepLearningFromForR/functions.R")
source("./DeepLearningFromForR/numerical_gradient.R")
source("./DeepLearningFromForR/utils.R")

# 2층 신경망에 초기값을 생성하는 함수입니다.
# 3층 신경망이 아니기에 가중치와 편향을 2개씩만 만들고 가중치에 고정값이 아닌 랜덤값을 부여하여 만듭니다. 
TwoLayerNet  <- function(input_size, hidden_size, output_size, weight_init_std = 0.01) {
  W1 <<- weight_init_std*matrix(rnorm(n = input_size*hidden_size), nrow = input_size, ncol = hidden_size)
  b1 <<- matrix(rep(0,hidden_size),nrow=1,ncol=hidden_size)
  W2 <<- weight_init_std*matrix(rnorm(n = hidden_size*output_size), nrow = hidden_size, ncol = output_size)
  b2 <<- matrix(rep(0,output_size),nrow=1,ncol=output_size)
  return(list(input_size, hidden_size, output_size,weight_init_std))
}

# 2층신경망의 계산 방법입니다. 
# 입력데이터를 인자로 받아 사용합니다. 
model.forward <- function(x){
  z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
  return(softmax(sweep((z1 %*% W2),2, b2,'+')))
}

# 손실 함수입니다. 
# 교차엔트로피 오차함수가 모델의 예측값과 정답을 통해 손실값을 계산합니다. 
loss <-function(x,t){
  return(cross_entropy_error(model.forward(x),t))
}

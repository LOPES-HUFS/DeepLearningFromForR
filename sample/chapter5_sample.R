library(rhdf5)
library(dslabs)

params <- list()

#init <-H5Fopen("../data/sample_weights.h5")
init <-H5Fopen("~/Sites/DeepLearningFromForR/data/sample_weights.h5")
model <- list(matrix(init$"W1",784,50), matrix(init$"b1",1,50), matrix(init$"W2",50,100), matrix(init$"b2",1,100), matrix(init$"W3",100,10), matrix(init$"b3",1,10))

params$W1 <- init$"W1"
params$W2 <- init$"W2"
params$b2 <- init$"b2"
params$b1 <- init$"b1"
params$b3 <- init$"b3"
params$W3 <- init$"W3"

Sigmoid <- function(x){
    out <- (1 / (1 + exp(-x)))
    return(list(out = out))
}

Affine.forward <- function(W, b, x){
    out <- sweep((x %*% W),2, b,'+')
    return(list(out = out, W = W, x = x))   
}

Affine.backward <- function(forward, dout){
  dx <- dout %*% t(forward$W)
  dW <- t(forward$x) %*% dout
  db <- matrix(colSums(dout), nrow=1)
  return(list(dx = dx, dW = dW, db = db))
}

predict <- function(x){
    Affine_1_layer <- Affine.forward(params$W1, params$b1, x)
    Sigmoid_1_layer <- Sigmoid(Affine_1_layer$out)
    Affine_2_layer <- Affine.forward(params$W2, params$b2, Sigmoid_1_layer$out)
    Sigmoid_2_layer <- Sigmoid(Affine_2_layer$out)
    Affine_3_layer <- Affine.forward(params$W3, params$b3, Sigmoid_2_layer$out)
    return(list(x = Affine_3_layer$out, Sigmoid_2.forward = Sigmoid_2_layer ,Affine_3.forward = Affine_3_layer, Affine_1.forward = Affine_1_layer, Affine_2.forward = Affine_2_layer, Sigmoid_1.forward = Sigmoid_1_layer))
}

model.evaluate <- function(x,t){
    y <- max.col(predict(x)$x)
    t <- max.col(t)
    #print(ifelse(y==t,paste(t-1,"숫자를 맞췄습니다"),paste(t-1,"숫자를 못 맞췄습니다")))
    accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
    return(accuracy)
}
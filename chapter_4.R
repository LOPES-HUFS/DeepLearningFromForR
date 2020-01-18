function_2 <- function(x){
    return(sum(colSums(x^2)))
}

numerical_gradient <- function(f,x){
    h <- 1e-4
    vec <- vector()
    temp <- rep(0,length(x))
    
    for(i in 1:length(x)){
        temp[i]<-(temp[i] + h)
        fxh1 <- function_2(x+temp)
        temp[i]<-(temp[i] - (2*h))
        fxh2<- function_2(x+temp)
        vec <- c(vec, (fxh1 - fxh2) / (2*h))
        temp[i]  <- 0
    }
    return(matrix(vec, nrow = nrow(x) ,ncol = ncol(x)))
}

gradient_descent <- function(f,init_x,lr = 0.01,step_num = 100){
    x <- init_x
    
    for(i in 1:step_num){
        grad <- numerical_gradient(f,x)
        x <- x-(lr*grad)
    }
    return(x)
}

simple.net_for_textBook <- setRefClass("simple.net",fields = list(W = "matrix"),methods = list(
    initialize = function(W = matrix(c(0.47355232, 0.9977393, 0.84668094,0.85557411, 0.03563661, 0.69422093), nrow = 2,byrow = T)){
        W <<- W
    },
    predict = function(x){
        return(x %*% W)
    },
    loss = function(x,t){
        z <- predict(x)
        y <- softmax(z)
        loss <- cross_entropy_error(y,t)
        return(loss)
        }
    )
)

softmax <- function(a){
    c <- max(a)
    sum_exp_a <- sum(exp(a - c))
    return(exp(a - c) / sum_exp_a) 
}

cross_entropy_error <- function(y, t) {
    delta <- 1e-7
    return(-sum(t*log(y+delta)))
}

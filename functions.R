numerical_gradient <- function(f,x){
    h <- 1e-4
    vec <- vector()
    temp <- rep(0,length(x))
    
    for(i in 1:length(x)){
        temp[i]<-(temp[i] + h)
        fxh1 <- f(x+temp)
        temp[i]<-(temp[i] - (2*h))
        fxh2<- f(x+temp)
        vec <- c(vec, (fxh1 - fxh2) / (2*h))
        temp[i]  <- 0
    }
    return(matrix(vec, nrow = nrow(x) ,ncol = ncol(x)))
}

making_one_hot_label <-function(t_label,nrow,ncol){
    data <- matrix(FALSE,nrow = nrow,ncol = ncol)
    t_index <- t_label+1
    for(i in 1:NROW(data)){
        data[i, t_index[i]] <- TRUE
    }
    return(data)
}

cross_entropy_error <- function(y, t){
    delta <- 1e-7
    batchsize <- dim(y)[1]
    return(-sum(t * log(y + delta))/batchsize)
}

cross_entropy_error_single <- function(y, t) {
    delta <- 1e-7
    return(-sum(t*log(y+delta)))
}

sigmoid <- function(x){
    return(1 / (1 + exp(-x)))
}

softmax_single <- function(a){
    c <- max(a)
    sum_exp_a <- sum(exp(a - c))
    return(exp(a - c) / sum_exp_a) 
}

softmax <- function(a){
    exp_a <- exp(a - apply(a,1,max))
    return(sweep(exp_a,1,rowSums(exp_a),"/"))
}

loss <-function(x,t){
    return(cross_entropy_error(predict(x),t))
}

model.evaluate <- function(x,t){
    y <- max.col(predict(x))
    t <- max.col(t)
    accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
    return(accuracy)
}
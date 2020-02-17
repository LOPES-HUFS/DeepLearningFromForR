numerical_gradient_layer <- function(f,x){
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

numerical_gradient_W1 <- function(f,x,t){
    h <- 1e-4
    vec <- matrix(0, nrow = nrow(W1) ,ncol = ncol(W1))
    for(i in 1:length(W1)){
        origin <- W1[i]
        W1[i] <<- (W1[i] + h)
        fxh1 <- f(x, t)
        W1[i] <<- (W1[i] - (2*h))
        fxh2 <- f(x, t)
        vec[i] <- (fxh1 - fxh2) / (2*h)
        W1[i] <<- origin
    }
    return(vec)
}
numerical_gradient_W2 <- function(f,x,t){
    h <- 1e-4
    vec <- matrix(0, nrow = nrow(W2) ,ncol = ncol(W2))
    for(i in 1:length(W2)){
        origin <- W2[i]
        W2[i] <<- (W2[i] + h)
        fxh1 <- f(x, t)
        W2[i] <<- (W2[i] - (2*h))
        fxh2 <- f(x, t)
        vec[i] <- (fxh1 - fxh2) / (2*h)
        W2[i] <<- origin
    }
    return(vec)
}
numerical_gradient_b1 <- function(f,x,t){
    h <- 1e-4
    vec <- matrix(0, nrow = nrow(b1) ,ncol = ncol(b1))
    for(i in 1:length(b1)){
        origin <- b1[i]
        b1[i] <<- (b1[i] + h)
        fxh1 <- f(x, t)
        b1[i] <<- (b1[i] - (2*h))
        fxh2 <- f(x, t)
        vec[i] <- (fxh1 - fxh2) / (2*h)
        b1[i] <<- origin
    }
    return(vec)
}

numerical_gradient_b2 <- function(f,x,t){
    h <- 1e-4
    vec <- matrix(0, nrow = nrow(b2) ,ncol = ncol(b2))
    for(i in 1:length(b2)){
        origin <- b2[i]
        b2[i] <<- (b2[i] + h)
        fxh1 <- f(x, t)
        b2[i] <<- (b2[i] - (2*h))
        fxh2 <- f(x, t)
        vec[i] <- (fxh1 - fxh2) / (2*h)
        b2[i] <<- origin
    }
    return(vec)
}

numerical_gradient_ <- function(f,x, t) {
    grads  <- list(W1 = numerical_gradient_W1(f,x,t), 
                   b1 = numerical_gradient_b1(f,x,t), 
                   W2 = numerical_gradient_W2(f,x,t), 
                   b2 = numerical_gradient_b2(f,x,t))
    return(grads)
}

numerical_gradient <- compiler::cmpfun(numerical_gradient)
softmax <- compiler::cmpfun(softmax)

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
    return(cross_entropy_error(model.forward(x),t))
}

model.forward <- function(x){
    z1 <- sigmoid(sweep((x %*% W1),2, b1,'+'))
    return(softmax(sweep((z1 %*% W2),2, b2,'+')))
}

model.evaluate <- function(x,t){
    y <- max.col(model.forward(x))
    t <- max.col(t)
    accuracy <- (sum(ifelse(y==t,1,0))) / dim(x)[1]
    return(accuracy)
}
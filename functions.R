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

identify_fun <- function(x){
    return(x)
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

softmax <- compiler::cmpfun(softmax)

padding <- function(input, num){
    input_nrow <- dim(input)[1]
    input_ncol <- dim(input)[2]
    n <- dim(input)[4]
    c <- dim(input)[3]
    temp <- array(0, dim = c((input_nrow + (2 * num)),(input_ncol + (2 * num)),c,n))
    temp_nrow <- nrow(temp)
    temp_ncol <- ncol(temp)
    temp[(1 + num):(temp_nrow - num), (1 + num):(temp_nrow - num),,] <- input
    return(temp)
}


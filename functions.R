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

Relu.forward <- function(x){
    mask <- x<=0
    out <- x
    out[mask] <- 0
    return(list(out = out, mask = mask))
}

Relu.backward <- function(forward, dout){
    dout[forward$mask] <- 0
    return(list(dx = dout))
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

SoftmaxWithLoss.forward <- function(x, t){
    y <- softmax(x)
    loss <- cross_entropy_error(y, t)
    return(list(loss = loss , y = y, t = t))
}

SoftmaxWithLoss.backward <- function(forward, dout=1){
    dx <- (forward$y - forward$t) / dim(forward$t)[1]
    return(list(dx = dx))
}

drop_out_single <- function(input_size, rate) {
    temp <- rep(TRUE, input_size)
    temp[sample(input_size, input_size * rate)] <- FALSE
    return(temp)
}
drop_out.forward <- function(input_size, hidden_size, rate = 0.5) {
    temp <- matrix(TRUE, nrow = hidden_size, ncol = input_size, byrow = TRUE)
    for(i in 1:hidden_size){
        temp[i,] <- drop_out_single(input_size, rate)
    }
    return(temp)
}


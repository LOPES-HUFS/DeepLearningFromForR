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

im2col <- function(input, filter_h, filter_w, stride, pad){   
    N <- dim(input)[4]
    c <- dim(input)[3]
    input_h <- dim(input)[2]
    input_w <- dim(input)[1]
    pad_temp <- padding(input, pad)
    output_r <- ((input_h + 2 * pad - filter_h) %/% stride) + 1 #OK
    
    output_c <- ((input_w + 2 * pad - filter_w) %/% stride) + 1 #OK
    result <- array(0, dim = c(filter_h, filter_w,output_r, output_c, c, N))
    for(i in 0:(filter_h-1)){
        i_max <- i + (stride * output_r)
        for(j in 0:(filter_w-1)){
            j_max <- j + (stride * output_c)
            result[j+1,i+1,,,,] <- pad_temp[seq(j+1, j_max, stride), seq(i+1, i_max, stride),,]
        }
    }
    reshape_result <- matrix(aperm(result,c(1,2,5,3,4,6)),output_r*output_c*N,byrow = T)
    return(reshape_result)
}

col2im <- function(col, input_data, filter_h, filter_w, stride, pad){
    n <- dim(input_data)[4]
    c <- dim(input_data)[3]
    h <- dim(input_data)[2]
    w <- dim(input_data)[1]
    out_h <- ((h + 2 * pad - filter_h) %/% stride) + 1
    out_w <- ((w + 2 * pad - filter_w) %/% stride) + 1
    col <- aperm(array(t(col),c(filter_h,filter_w,c,out_h,out_w,n)),c(4,5,1,2,3,6))
    result <- array(0,c(h + 2*pad + stride - 1, w + 2*pad + stride - 1,c,n ))
    for(i in 1:filter_h){
        i_max <- (i +stride * out_h)-1
        for(j in 1:filter_w){
            j_max <- (j + stride * out_w)-1
            result[seq(i, i_max, stride),seq(j, j_max, stride),,] <- result[seq(i, i_max, stride),seq(j, j_max, stride),,]+col[,,i,j,,]
            
        }
    }
    new_result <- result[(1+pad):(h + pad), (1+pad):(w + pad),,]
    new_result <- array(new_result,c(dim(new_result)[1],dim(new_result)[2],c,n))
    return(new_result)
}

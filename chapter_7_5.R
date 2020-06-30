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
  input_h <- dim(input)[1]
  input_w <- dim(input)[2]
  temp <- padding(input, pad)
  output_r <- ((input_h + 2 * pad - filter_h) %/% stride) + 1 #OK
  output_c <- ((input_w + 2 * pad - filter_w) %/% stride) + 1 #OK
  result <- array(0, dim = c(filter_h, filter_w,output_r, output_c, c, N))
  for(i in 0:(filter_h-1)){
    i_max <- i + (stride * output_r)
    for(j in 0:(filter_w-1)){
      j_max <- j + (stride * output_c)
      result[j+1,i+1,,,,] <- temp[seq(j+1, j_max, stride), seq(i+1, i_max, stride),,]
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
  col <- aperm(array(t(col),c(filter_h,filter_w,c,h,w,n)),c(4,5,1,2,3,6))
  result <- array(0,c(h + 2*pad + stride - 1, w + 2*pad + stride - 1,c,n))
  for(i in 0:(filter_h-1)){
    i_max <- i + (stride * out_h)
    for(j in 0:(filter_w-1)){
      j_max <- j + (stride * out_w)
      result[seq(j+1, j_max, stride), seq(i+1, i_max, stride),,] <- result[seq(j+1, j_max, stride), seq(i+1, i_max, stride),,]+col[j+1,i+1,,,,]
    }
  }
  return(result[pad:H + pad, pad:W + pad,,])
}

convolution_forward <- function(x,W,b,stride,pad){
  fn <- dim(W)[4]
  fc <- dim(W)[3]
  fh <- dim(W)[2]
  fw <- dim(W)[1]
  n <- dim(x)[4]
  c <- dim(x)[3]
  h <- dim(x)[2]
  w <- dim(x)[1]
  out_h <- ((h + 2 * pad - fh) %/% stride) + 1
  out_w <- ((w + 2 * pad - fw) %/% stride) + 1
  col <- im2col(x, fh, fw, stride, pad)
  col_w <- t(matrix(aperm(W,c(3,4,1,2)),fn*fc,fh*fw))
  out <- sweep(col%*%col_w,2,b,"+") 
  new_out <- aperm(array(out,c(out_h,out_w,n,fn)),c(1,2,4,3))
  return(list(out=new_out,x=x,col=col,col_w=col_w))
}

convolution_backward <- function(dout,W,stride=1,pad=0){
  fn <- dim(W)[4]
  fc <- dim(W)[3]
  fh <- dim(W)[2]
  fw <- dim(W)[1]
  new_dout <- matrix(aperm(dout$out,c(1,2,4,3)),ncol=fn)
  db <- colSums(new_dout)
  dW <- t(dout$col)%*%new_dout
  dW <- array(dW,c(fw,fh,fc,fn))
  dcol <- new_dout%*%t(dout$col_w)
  dx <- col2im(dcol, dout$x, fh, fw,stride, pad)
  return(dx)
}

pooling.forward <- function(x, pool_h, pool_w, stride, pad){
  n <- dim(x)[4]
  c <- dim(x)[3]
  h <- dim(x)[2]
  w <- dim(x)[1]
  out_h <- ((h + 2 * pad - pool_h) %/% stride) + 1
  out_w <- ((w + 2 * pad - pool_w) %/% stride) + 1
  col_im <- im2col(x, pool_h, pool_w, stride, pad)
  col <- matrix(t(col_im), ncol=pool_h*pool_w,byrow=T)
  arg_max <- matrix(0,nrow=NROW(col))
  out <- matrix(0,nrow=NROW(col))
  for(i in 1:NROW(col)){
    if(sum(col[i,])==0){
      arg_max[i,] <- 1
    }
    else{
      arg_max[i,] <- which.max(col[i,])
    }
    out[i,] <- max(col[i,])
  }
  new_out <- aperm(array(out,c(c,out_h,out_w,n)),c(2,3,1,4))
  return(list(out=new_out,x=x,argmax=arg_max,col=col))
  
}

flatten_forward <- function(x){
  n <- dim(x)[4]
  c <- dim(x)[3]
  h <- dim(x)[2]
  w <- dim(x)[1]
  out <- matrix(0, nrow = n, ncol = h*w*c)
  mask <- dim(x)
  for(i in 1:n){
    data <- x[,,,i]
    temp <- matrix(data, nrow = 1, ncol = h*w*c)
    out[i,] <- temp
  }
  return(list(out = out , mask = mask, x=x))
  
}
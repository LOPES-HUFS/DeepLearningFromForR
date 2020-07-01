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
  col <- aperm(array(t(col),c(filter_h,filter_w,c,out_h,out_w,n)),c(4,5,1,2,3,6))
  result <- array(0,c(h + 2*pad + stride - 1, w + 2*pad + stride - 1,c,n ))
  for(i in 1:filter_h){
    i_max <- (i +stride * out_h)-1
    for(j in 1:filter_w){
      j_max <- (j + stride * out_w)-1
      result[seq(j, j_max, stride),seq(i, i_max, stride),,] <- result[seq(j, j_max, stride),seq(i, i_max, stride),,]+col[,,j,i,,] 
    }
  }
  new_result <- result[(1+pad):(h + pad), (1+pad):(w + pad),,]
  new_result_reshape <- array(new_result,c(dim(new_result)[1],dim(new_result)[2],c,n))
  return(new_result_reshape)
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

pooling.backward <- function(pool_forward,dout,pool_h,pool_w,stride,pad){
  dout <- aperm(pool_forward$out,c(3,1,2,4))
  pool_size <- pool_h * pool_w
  dmax <- matrix(0,nrow = length(dout), ncol = pool_size)
  for(i in 1:dim(pool_forward$argmax)[1]){
    dmax[i,array(c(pool_forward$argmax))[i]] <- array(c(dout))[i]
  }
  #dmax <- array(t(dmax),c(pool_size,dim(dout)))
  dcol <- dmax
  dx <- col2im(dcol,pool_forward$x,pool_h,pool_w,stride,pad)
  return(dx)
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
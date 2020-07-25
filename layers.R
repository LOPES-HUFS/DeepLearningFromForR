source("./functions.R")
source("./utils.R")

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
drop_out <- function(x,rate = 0.5){
  temp <- matrix(TRUE, nrow = NROW(x), ncol = NCOL(x), byrow = TRUE)
  for(i in 1:NROW(x)){
    temp[i,] <- drop_out_single(NCOL(x), rate)
  }
  return(temp)
}

drop_out.forward <- function(x, rate) {
  temp <- drop_out(x,rate)
  return(list(out = x*temp, mask = temp))
}

drop_out.backward<- function(forward,dout) {
  dout <- dout*forward$mask
  return(list(dx = dout))
}


convolution.forward <- function(x,W,b,stride,pad){
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
  out <- aperm(array(out,c(out_h,out_w,n,fn)),c(2,1,4,3))
  return(list(out=out,x=x,col=col,col_w=col_w,W=W))
}

convolution.backward <- function(convolution_forward,dout,stride=1,pad=0){
  fn <- dim(convolution_forward$W)[4]
  fc <- dim(convolution_forward$W)[3]
  fh <- dim(convolution_forward$W)[2]
  fw <- dim(convolution_forward$W)[1]
  new_dout <- matrix(aperm(dout,c(1,2,4,3)),ncol=fn)
  db <- colSums(new_dout)
  dW <- t(convolution_forward$col)%*%new_dout
  dW <- array(dW,c(fw,fh,fc,fn))
  dcol <- new_dout%*%t(convolution_forward$col_w)
  dx <- col2im(dcol, dim(convolution_forward$x), fh, fw,stride, pad)
  return(list(dx=dx,dW=dW,db=db))
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
  arg_max <- apply(col,1,which.max)
  out <- apply(col,1,max)
  new_out <- aperm(array(out,c(c,out_h,out_w,n)),c(3,2,1,4))
  return(list(new_out=new_out,x=x,argmax=arg_max,out=out))
  
}

pooling.backward <- function(pool_forward,dout,pool_h,pool_w,stride,pad){
  dout <- aperm(dout,c(3,1,2,4))
  pool_size <- pool_h * pool_w
  dmax <- matrix(0,nrow = length(pool_forward$argmax), ncol = pool_size)
  prebound <- cbind(1:length(pool_forward$argmax),pool_forward$argmax)
  dmax[prebound] <- c(dout)
  dmax <- array(t(dmax),dim = c(pool_size,dim(dout)))
  dcol <- matrix(dmax,dim(dmax)[3]*dim(dmax)[4]*dim(dmax)[5],dim(dmax)[1]*dim(dmax)[2],T)
  dx <- col2im(dcol,dim(pool_forward$x),pool_h,pool_w,stride,pad)
  return(dx)
}

flatten.forward <- function(x){
  mask <- dim(x)
  out <- t(matrix(x,nrow=dim(x)[1]*dim(x)[2]*dim(x)[3],ncol=dim(x)[4]))
  return(list(out = out , mask = mask))
}

flatten.backward <- function(flatten_forward,dout){
  temp_out <- array(t(dout), dim = flatten_forward$mask)
  return(aperm(temp_out, c(2, 1, 3, 4)))
}

forward <- function(name, x, params=NA){
  if(all(is.na(params)) == TRUE){
    if(name=="Flatten"){return(flatten.forward(x))}
    else if(name =="ReLU"){return(Relu.forward(x))}
    else{return("name is False")}
  }
  else{
    if(name=="Affine"){
      return(Affine.forward(params[["W"]],params[["b"]], x=x))}
    else if(name == "SoftmaxWithLoss"){
      return(SoftmaxWithLoss.forward(x,params[["t"]]))}
    else if(name=="Convolution"){
      return(convolution.forward(x=x,W=params[["W"]],b=params[["b"]],
                                 stride = params[["stride"]],pad = params[["pad"]]))}
    else if(name == "Pooling"){
      return(pooling.forward(x, params[["pool_h"]], params[["pool_w"]], 
                             params[["stride"]], params[["pad"]]))}
    else if(name == "Drop_out"){
      return(drop_out.forward(x,params[["rate"]]))
    }
    else{
      return("name is False")
    }
  }
}

backward <- function(name,forward,dout,params=NA){
  if(all(is.na(params))==FALSE){
    return(pooling.backward(forward,dout,params[["pool_h"]],
                            params[["pool_w"]],params[["stride"]],params[["pad"]]))
  }
  else{
    if(name == "ReLU"){return(Relu.backward(forward,dout))}
    else if(name == "Affine"){return(Affine.backward(forward,dout))}
    else if(name == "SoftmaxWithLoss"){return(SoftmaxWithLoss.backward(forward,dout))}
    else if(name == "Convolution"){return(convolution.backward(forward,dout))}
    else if(name == "Flatten"){return(flatten.backward(forward,dout))}
    else if(name == "Drop_out"){return(drop_out.backward(forward,dout))}
    else{return("name is False")}
  }
}
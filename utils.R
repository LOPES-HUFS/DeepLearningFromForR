making_one_hot_label <-function(t_label,nrow,ncol){
  #use example: making_one_hot_label(t_train,60000,10)
  data <- matrix(FALSE,nrow = nrow,ncol = ncol)
  t_index <- t_label+1
  for(i in 1:NROW(data)){
    data[i, t_index[i]] <- TRUE
  }
  return(data)
}

draw_image <- function(x){
  #use example: draw_image(x_test[5,])
  return(image(1:28, 1:28, matrix(x, nrow=28)[ , 28:1], col = gray(seq(0, 1, 0.05)), xlab = "", ylab=""))
}

get_data<- function(tensor=FALSE){
  mnist<-read_mnist()
  x_train<-mnist$train$images
  t_train<-mnist$train$labels
  x_test<-mnist$test$images
  t_test<-mnist$test$labels
  if(tensor==TRUE){
    x_train <- array(t(x_train),c(28,28,1,60000))
    x_test <- array(t(x_test),c(28,28,1,10000))
  }
  x_train_normalize <- x_train/255
  x_test_normalize <- x_test/255
  
  return(list(x_train=x_train_normalize,x_test=x_test_normalize,t_train=t_train,t_test=t_test))
}

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
  result <- array(0, dim = c(output_r, output_c,filter_h, filter_w, c, N))
  for(i in 1:(filter_h)){
    i_max <- (i-1) + (stride * output_r)
    for(j in 1:(filter_w)){
      j_max <- (j-1) + (stride * output_c)
      result[,,j,i,,] <- pad_temp[seq(j, j_max, stride), seq(i, i_max, stride),,]
    }
  }
  reshape_result <- matrix(aperm(result,c(4,3,5,2,1,6)),output_r*output_c*N,byrow = T)
  return(reshape_result)
}

col2im <- function(col, input_dim, filter_h, filter_w, stride, pad){
  n <- input_dim[4]
  c <- input_dim[3]
  h <- input_dim[2]
  w <- input_dim[1]
  out_h <- ((h + 2 * pad - filter_h) %/% stride) + 1
  out_w <- ((w + 2 * pad - filter_w) %/% stride) + 1
  col <- aperm(array(t(col),c(filter_h,filter_w,c,out_h,out_w,n)),c(5,4,1,2,3,6))
  newcol<-aperm(col,c(1,2,4,3,5,6))
  result <- array(0,c(h + 2*pad + stride - 1, w + 2*pad + stride - 1,c,n))
  for(i in 1:filter_h){
    i_max <- (i +stride * out_h)-1
    for(j in 1:filter_w){
      j_max <- (j + stride * out_w)-1
      result[seq(i, i_max, stride),seq(j, j_max, stride),,] <- result[seq(i, i_max, stride),seq(j, j_max, stride),,]+newcol[,,i,j,,]
      
    }
  }
  new_result <- result[(1+pad):(h + pad), (1+pad):(w + pad),,]
  new_result_reshape <- array(new_result,c(dim(new_result)[1],dim(new_result)[2],c,n))
  return(new_result_reshape)
}

  
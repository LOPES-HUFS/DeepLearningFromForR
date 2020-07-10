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
    x_train <- array(x_train,28,28,1,60000)
    x_test <- array(x_test,28,28,1,60000)
  }
  x_train_normalize <- x_train/255
  x_test_normalize <- x_test/255
  
  return(list(x_train=x_train_normalize,x_test=x_test_normalize,t_train=t_train,t_test=t_test))
}
  
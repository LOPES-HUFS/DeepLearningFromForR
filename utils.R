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


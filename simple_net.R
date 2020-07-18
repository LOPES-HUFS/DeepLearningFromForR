source("./functions.R")
source("./utils.R")
source("./optimizer.R")
source("./cnn_functions.R")

make_parameter <- function(input_dim,filter_size,filter_num,pad,stride,hidden_size,output_size){
  return(list(input_dim=input_dim,filter_size=filter_size,filter_num=filter_num,
              pad=pad,stride=stride,hidden_size=hidden_size,output_size=output_size,
              weight_init_std=0.01))
}

simple_net_params <- function(params){
  input_size <- params[["input_dim"]][1]
  conv_output_size <- (input_size - params[["filter_size"]] + 2*params[["pad"]]) / params[["stride"]] + 1
  pool_output_size <- as.numeric(params[["filter_num"]] * (conv_output_size/2)^2)
  
  W1 <- params[["weight_init_std"]]*array(rnorm(n = params[["input_dim"]][3]*params[["filter_size"]]^2*params[["filter_num"]]),dim = c(params[["filter_size"]],params[["filter_size"]],params[["input_dim"]][3],params[["filter_num"]]))
  b1 <- matrix(rep(0,params[["filter_num"]]),nrow=1,ncol=params[["filter_num"]])
  W2 <- params[["weight_init_std"]]*matrix(rnorm(n = pool_output_size*params[["hidden_size"]]), 
                                           nrow = pool_output_size, ncol = params[["hidden_size"]])
  b2 <- matrix(rep(0,params[["hidden_size"]]),nrow=1,ncol=params[["hidden_size"]])
  W3 <- params[["weight_init_std"]] * matrix(rnorm(params[["hidden_size"]]*params[["output_size"]]),
                                             nrow=params[["hidden_size"]],ncol=params[["output_size"]])
  b3 <- matrix(rep(0,params[["output_size"]]),nrow=1,ncol=params[["output_size"]])
  network <- list(W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3)
  return(network)
}

init <- function(tensor){
  mnist_data <- get_data(tensor)
  #손글씨 데이터
  x_train_normalize <<- mnist_data$x_train 
  x_test_normalize <<- mnist_data$x_test
  #정답 레이블
  t_train_onehotlabel <<- making_one_hot_label(mnist_data$t_train,60000, 10)
  t_test_onehotlabel <<- making_one_hot_label(mnist_data$t_test,10000, 10)
} 

pool_params <- list(pool_h=2, pool_w=2, stride=2, pad=0)
model.forward <- function(network, x){
  conv_params <- list(W = network$W1,b = network$b1, stride = 1, pad = 0)
  affine_params_1 <- list(W = network$W2,b = network$b2)
  affine_params_2 <- list(W = network$W3,b = network$b3)
  
  conv_temp <- forward("Convolution",x,conv_params)
  relu_temp_1 <- forward("ReLU",conv_temp$out)
  pool_temp <- forward("Pooling",relu_temp_1$out,pool_params)
  flatten_temp <- forward("Flatten",pool_temp$out)
  affine_temp_1 <- forward("Affine",flatten_temp$out,affine_params_1)
  relu_temp_2 <- forward("ReLU",affine_temp_1$out)
  affine_temp_2 <- forward("Affine",relu_temp_2$out,affine_params_2)
  return(list(x = affine_temp_2$out, Affine_1.forward = affine_temp_1, Affine_2.forward = affine_temp_2, Relu_1.forward = relu_temp_1,Relu_2.forward = relu_temp_2,conv_temp = conv_temp,pool_temp=pool_temp,flatten = flatten_temp))
}

loss <- function(model.forward, network, x, t){
  softmax_params <- list(t = t)
  temp <- model.forward(network, x)
  y <- temp$x
  last_layer.forward <- forward("SoftmaxWithLoss",y,softmax_params) # 이부분 뭔가 이상 
  return(list(loss = last_layer.forward$loss, softmax = last_layer.forward, predict =  temp))
}

model.backward <- function(model.forward, network, x, t) {
  # 순전파
  loss_temp <- loss_test(model.forward, network, x, t)
  # 역전파
  dout <- 1
  dout <- backward("SoftmaxWithLoss",loss_temp$softmax,dout)
  dout1 <- backward("Affine",loss_temp$predict$Affine_2.forward,dout$dx)
  dout2 <- backward("ReLU",loss_temp$predict$Relu_2.forward,dout1$dx)
  dout3 <- backward("Affine",loss_temp$predict$Affine_1.forward,dout2$dx)
  dout_3 <- backward("Flatten",loss_temp$predict$flatten,dout3$dx)
  dx <- backward("Pooling",loss_temp$predict$pool_temp,dout_3,pool_params)
  dx1 <- backward("ReLU",loss_temp$predict$Relu_1.forward,dx)
  dx2 <- backward("Convolution",loss_temp$predict$conv_temp,dx1$dx)
  grads  <- list(W1  =  dx2$dW, b1  =  dx2$db, W2  =  dout3$dW, b2  =  dout3$db, W3 = dout1$dW, b3 = dout1$db)
  return(grads)
}

model.evaluate <- function(model, network, x, t){
  
  temp <- model(network, x)
  y <- max.col(temp$x)
  t <- max.col(t)
  accuracy <- ifelse(NROW(dim(x))==2,sum(ifelse(y == t,1,0)) / dim(x)[1], sum(ifelse(y == t,1,0)) / dim(x)[4])
  return(accuracy)
}

model.train <- function(train_x,train_t, test_x, test_t, batch_size, iters_num, optimizer_name, debug=FALSE){
  train_size <- dim(train_x)[4]
  iter_per_epoch <- max(train_size / batch_size)
  
  params <- make_parameter(c(28,28,1),5,30,0,1,100,10)
  network <- simple_net_params(params = params)  
  for(i in 1:iters_num){
    batch_mask <- sample(train_size ,batch_size)
    x_batch <- train_x[,,,batch_mask]
    x_batch <- array(x_batch,c(28,28,1,100))
    t_batch <- train_t[batch_mask,]
    
    gradient <- gradient_test(model.forward_test,network, x_batch, t_batch)
    network <- get_optimizer(network, gradient, optimizer_name)
    if(debug=TRUE){
      if(i %% 600 ==0){
        train_loss_list <- rbind(train_loss_list,loss_test(model.forward=model.forward_test, network = network, x_batch, t_batch)$loss)
        test_acc <- rbind(test_acc,model.evaluate(model.forward_test, network, x_test_normalize, t_test_onehotlabel))
      }
    }
    
  }
  return(list(loss=train_loss_list,test_acc=test_acc))
}


train_loss_list <- data.frame(loss_value = 0)
test_acc <- data.frame(acc = 0)
init(TRUE)
params <- make_parameter(c(28,28,1),5,30,0,1,100,10)
network <- simple_net_params(params = params)  

model_temp <- model.train(train_x = x_train_normalize,train_t = t_train_onehotlabel,x_test_normalize,t_test_onehotlabel,batch_size = 100,iters_num = 3000,optimizer_name = "adam",debug = TRUE)
# 아래와 같이 model.forward를 만들어
#
# model.forward <- function(x){
#    Affine_1_layer <- Affine.forward(params$W1, params$b1, x)
#    Relu_1_layer <- Relu.forward(Affine_1_layer$out)
#    Affine_2_layer <- Affine.forward(params$W2, params$b2, Relu_1_layer$out)
#    return(Affine_2_layer$out)
#}
#
# 모델을 평가할 때, 아래와 같이 사용하면 된다.
#
# model.evaluate(model.forward, x_train_normalize, t_train_onehotlabel)
# model.evaluate(model.forward, x_test_normalize, t_test_onehotlabel)

model.evaluate <- function(model.forward, x, t){
    y <- max.col(model.forward(x))
    t <- max.col(t)
    accuracy <- (sum(ifelse(y == t, 1, 0))) / dim(x)[1]
    return(accuracy)
}

# 특정 숫자를 맞출 수 있는지 아래와 같이 사용하면 됩니다.

# model.predict(model.forward, x_train_normalize[2,])

model.predict <- function(model.forward, x){
    return(softmax(model.forward(x)))
}

model.train_set_validator <- function(x_train, y_train){
    return(dim(x_train)[2] != dim(x_test)[2])
}

model.train <- function(batch_size, iters_num, learning_rate,
                        network, x_train, y_train, x_test, y_test,
                        debug = FALSE){
    
    train_size <- dim(x_train)[1]
    
    iter_per_epoch <- max(train_size / batch_size)
    
    for(i in 1:iters_num){
        batch_mask <- sample(train_size ,batch_size)
        x_batch <- x_train[batch_mask,]
        t_batch <- y_train[batch_mask,]
        
        grad <- gradient(x_batch, t_batch)
        
        #update weights and biases using SGD
        network <<- sgd.update(network, grad, lr = learning_rate)
        
        if(debug == TRUE){
            if(i %% iter_per_epoch == 0){
                train_acc <- model.evaluate.backward(x_train_normalize, t_train_onehotlabel)
                test_acc <- model.evaluate.backward(x_test_normalize, t_test_onehotlabel)
                print(c(train_acc, test_acc))
            }
        }
    }

    train_accuracy = model.evaluate.backward(x_train_normalize, t_train_onehotlabel)
    test_accuracy = model.evaluate.backward(x_test_normalize, t_test_onehotlabel)
    return(c(train_accuracy, test_accuracy))
}
sigmoid.forward <- function(x){
  out <- 1 / (1+exp(-x))
  return(list(out=out))
}

sigmoid.backward <- function(forward,dout){
  dx <- dout*(1 - forward$out)*forward$out
  return(list(dx=dx))
}
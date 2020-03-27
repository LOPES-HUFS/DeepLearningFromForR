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
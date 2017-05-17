
pm.bidiag <- function(x)
{
  m <- nrow(x)
  n <- ncol(x)
  iterations <- min(n - (n == m), m)
  
  H <- diag(m)
  S <- diag(n)
  for (i in 1:iterations)
  {
    h <- colHouseholderMatrix(x[,i], i)
    H <- H %*% h
    x <- h %*% x
    
    if (i != iterations)
    {
      s <- rowHouseholderMatrix(x[i,], i)
      S <- S %*% s
      x <- x %*% s
    }
  }
  return(list(
    B = x * .biDiag(m, n),
    P = H,
    S = S
  ))
}

colHouseholderMatrix <- function(x, i)
{
  m <- length(x)
  
  # Replaces first i-1 elements with 0s
  x <- c(rep(0, i-1), x[i:m])
  y <- c(rep(0, i-1), -pm.householderSign(x[i]) * pm.vectorNorm(x), rep(0, m - i))
  
  w <- (x - y) / pm.vectorNorm(x - y)
  if (any(!is.finite(w)))
    w <- rep(0, m)
  
  return(diag(m) - (2 * (w %*% t(w))))
}

pm.householderSign <- function(x)
{
  if (x >= 0)
    return(1)
  else
    return(-1)
}

rowHouseholderMatrix <- function(x, i)
{
  m <- length(x)
  
  x <- c(rep(0, i), x[(i+1):m])
  y <- c(rep(0, i), pm.vectorNorm(x), rep(0, m - i - 1))
  
  w <- (x - y) / pm.vectorNorm(x - y)
  if (any(!is.finite(w)))
    w <- rep(0, m)
  
  return(diag(m) - (2 * (w %*% t(w))))
}

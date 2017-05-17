
#' Transforms a symmetrical matrix into tridiagonal form using the householder method.
#' 
#' @title pm.tridiag
#' @param A - symmetrical matrix to be converted to tridiagonal form.
#' @return Transformed symmetrical tridiagonal matrix of A.
#' 
#' @author Jesse Bannon
#' @rdname pm.tridiag
#' 
pm.tridiag <- function(A)
{
  m <- ncol(A)
  n <- nrow(A)
  for (i in 1:(n-2))
  {
    H <- .householderMatrix(A[,i], i)
    A <- H %*% A %*% H
  }
  return(A * .tri(n))
}

#' Helper function for pm.tridiag
#' Returns a tridiagonal n x n matrix with its tridiag elements initialized as 1's.
#' 
#' @title .tri
#' @param n - row and column length of the tridiagonal matrix output.
#' @return n x n tridiagonal matrix initialized as all 1's.
#' 
#' @author Jesse Bannon
#' @rdname .tri
#' 
.tri <- function(n)
{
  temp <- diag(n)
  temp1 <- diag(n+1)
  
  return(temp + temp1[1:n, 2:(n+1)] + temp1[2:(n+1), 1:n])
}

#' Helper function for pm.tridiag
#' Returns a householder matrix for the given vector x from column i of some matrix.
#' 
#' @title .householderMatrix
#' @param x - ith column of the matrix
#' @param i - column number of the matrix
#' 
#' @author Jesse Bannon
#' @rdname .householderMatrix
#' 
.householderMatrix <- function(x, i)
{
  m <- length(x)
  s <- pm.vectorNorm(x[(i+1):m])
  if (s == 0)
    householderMatrix(x, i+1)
  
  SG <- pm.householderSign(x[i+1])
  
  z <- 1/2 * (1 + SG*x[i+1]/s)
  
  v <- c(rep(0, i), sqrt(z), SG*x[(i+2):m]/(2 * sqrt(z) * s))
  return(diag(m) - (2 * (v %*% t(v))))
}

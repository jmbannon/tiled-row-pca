#------------------------------------------------------------------------------
# PLEASE NOTE:
# This file contains unfinished functions pertaining to solving for
# eigen decompositions. Do not use this in production.
#------------------------------------------------------------------------------

#' Computes eigenvalues and eigenvectors using naive iterative QR decomposition.
#' 
#' @title pm_eigen
#' @param x - a numeric matrix whos spectral decomposition is to be computed.
#' @param symmetric - logical; whether it is symmetric or not. ONLY SYMMETRIC MATRICES ARE SUPPORTED RIGHT NOW
#' @param only.values - logical; if TRUE, only the eigenvalues are computed and returned, otherwise
#' both eigenvalues and eigenvectors are returned.
#' @param EISPACK - logical; Defunct and ignored.
#' @return values - Eigenvalues. vectors - Eigenvectors.
#' @author Jesse Bannon
#' @rdname pm_eigen
#' 
pm.eigen <- function(x, symmetric = FALSE, only.values = FALSE, EISPACK = FALSE)
{
  n <- nrow(x)
  if (!is.matrix(x) || n != ncol(x))
  {
    stop("Non-square matrix in 'eigen'")
  }
  if (!symmetric)
  {
    stop("Non-symmetric matrices not supported yet.")
  }
  
  pQ <- if (only.values) NULL else diag(n)
  
  # Used to compare eigenvalue differences between each iteration.
  eigPrev <- vector("numeric", length = n)
  eigNext <- vector("numeric", length = n)
  maxDiff <- 1
  
  epsilon <- 1.0e-04
  maxIterations <- 1000
  i <- 1
  
  while (i < maxIterations && maxDiff > epsilon)
  {
    qrX <- pm.QR(x)
    x <- qrX$R %*% qrX$Q
    
    # Eigenvalues lie on the diagnol of x
    if (i == 1)
    {
      eigPrev <- diag(x)
    }
    else
    {
      eigPrev <- eigNext
      eigNext <- diag(x)
      
      # Gets the maximum difference between all previous and current eigenvalues.
      maxDiff <- max(abs(eigNext - eigPrev))
    }
    
    if (!only.values)
    {
      pQ <- pQ %*% qrX$Q
    }
    i <- i+1
  }
  print(sprintf("total iterations: %d \n", i))
  
  return(list(
    values=eigNext,
    vectors=pQ
  ))
}


#' Solves for eigenpairs using Cuppen's divide and conquer algorithm.
#' NOTE: This function is unfinished. Need to solve the secular equation to obtain eigenvalues.
#' 
#' @title pm.eigen.dc
#' @param x - Square matrix to solve eigen decomposition.
#' @param symmetric - logical; whether it is symmetric or not. ONLY SYMMETRIC MATRICES ARE SUPPORTED RIGHT NOW
#' @param only.values - logical; if TRUE, only the eigenvalues are computed and returned, otherwise
#' both eigenvalues and eigenvectors are returned.
#' @return values - Eigenvalues. vectors - Eigenvectors.
#' 
#' @author Jesse Bannon
#' @rdname pm.eigen.dc
#' 
pm.eigen.dc <- function(x, symmetric = TRUE, only.values = FALSE)
{
  n <- nrow(x)
  if (!is.matrix(x) || n != ncol(x))
  {
    stop("Non-square matrix in 'eigen'")
  }
  if (!symmetric)
  {
    stop("Non-symmetric matrices not supported yet.")
  }
  
  x <- pm.tridiag(x)
  splitT <- splitTri(x)
  
  beta <- splitT$beta
  T1 <- splitT$T1
  T2 <- splitT$T2
  
  eig1 <- pm_eigen(T1, symmetric = TRUE)
  eig2 <- pm_eigen(T2, symmetric = TRUE)
  
  Q1 <- eig1$vectors
  Q2 <- eig2$vectors
  
  T0 <- mergeMatrices(T1, T2)
  M1 <- mergeMatrices(Q1, Q2)
  M2 <- mergeMatrices(t(Q1), t(Q2))
  
  Delements <- c(eig1$values, eig2$values)
  Dorder <- order(Delements)
  

  D <- diag(Delements[Dorder])
  v <- c(Q1[nrow(Q1),], Q2[,1])[Dorder]

  # Need to solve the secular equation for elements in diag(D)
  
  # This is D + p*z*zT
  # (D + beta * (v %*% t(v)))
  
  # Should equal eigenvalues of x
  # print(eigen(D + beta * (v %*% t(v)))$values)
  
  return(NULL)
}

#' Helper function for pm.eigen.dc. Splits a tridiagonal matrix into two tridiagonals.
#' 
#' @param x - A tridiagonal matrix.
#' @return T1 - Tridiagonal from upper-left matrix. T2 - Tridiagonal from lower-right matrix.
#' Beta - Value above the split element (only value from x not in T1 or T2).
#' 
#' @author Jesse Bannon
#' @rdname splitTri
#' 
splitTri <- function(x)
{
  n <- nrow(x)
  
  T1Size <- as.integer(n/2) + (n %% 2)
  T2Size <- n - T1Size
  
  beta <- x[T1Size, T1Size+1]
  T1 <- as.matrix(x[1:T1Size, 1:T1Size], nrow = T1Size, ncol = T1Size)
  T2 <- as.matrix(x[(T1Size+1):n, (T1Size+1):n], nrow = T2Size, ncol = T2Size)
  
  T1[T1Size, T1Size] <- T1[T1Size, T1Size] - beta
  T2[1,1] <- T2[1,1] - beta
  #u <- c(rep(0, T1Size-1), 1, 1, rep(0, T2Size-1))
  #u <- beta * (u %*% t(u))
  #u <- beta * u
  
  return(list(
    T1 = T1,
    T2 = T2,
    beta = beta
  #  u = u
  ))
}

#' Helper function for pm.eigen.dc
#' Merges two matrices; x1 in the upper-left portion and x2 in the lower-right portion.
#' 
#' @title mergeMatrices
#' @param x1 - Matrix to return in upper-left corner.
#' @param x2 - Matrix to return in lower-right corner.
#' @return Matrix where x1 is in the upper-left corner and x2 is in the lower-right corner. Other corners are zeroed out.
#' 
#' @author Jesse Bannon
#' @rdname mergeMatrices
#' 
mergeMatrices <- function(x1, x2)
{
  n1 <- nrow(x1)
  n2 <- nrow(x2)
  
  
  toRet1 <- sapply(1:n1, FUN = function(i) { c(x1[,i], rep(0, n2))})
  toRet2 <- sapply(1:n2, FUN = function(i) { c(rep(0, n1), x2[,i])})
  
  return(cbind(toRet1, toRet2))
}

#' Helper function for pm.eigen.dc
#' Solves eigendecomposition on a 2x2 or 1x1 matrix.
#' 
#' @param x - 2x2 or 1x1 matrix to solve eigendecomposition of.
#' @return values - Eigenvalues. vectors - Eigenvectors.
#' 
#' @author Jesse Bannon
#' @rdname eigSmall
#' 
eigSmall <- function(x)
{
  if (nrow(x) == 2)
  {
    b <- -(x[1,1] + x[2,2])
    c <- ((x[1,1] * x[2,2]) - (x[1,2] * x[2,1]))
    eigValues <- polyRoots(b, c)
    
    if (x[2,1] != 0)
      eigVectors <- matrix(c(eigValues[1] - x[2,2], x[2,1], eigValues[2] - x[2,2], x[2,1]), nrow = 2)
    else if (x[1,2] != 0)
      eigVectors <- matrix(c(x[1,2], eigValues[1] - x[1,1], x[2,1], eigValues[2] - x[1,1]), nrow = 2)
    else
      eigVectors <- matric(c(1,0,0,1), nrow=2)
    
    eigVectors <- apply(eigVectors, 2, FUN = function(x) { x / pm.vectorNorm(x) })
    return(list(
      values = eigValues,
      vectors = eigVectors
    ))
  }
  else # nrow(x) == 1
  {
    return(list(
      values = x[1,1],
      vectors = matrix(1, nrow = 1)
    ))
  }
}

#' Helper function for eigSmall.
#' Solves the roots of a simple polynomial equation.
#' 
#' @param b - b in the poly-root equation.
#' @param c - c in the poly-root equation.
#' @return Roots of the poly equation.
#' 
#' @author Jesse Bannon
#' @rdname polyRoots
#' 
polyRoots <- function(b, c)
{
  root <- sqrt(b^2 - (4*c))
  return(c(((-b + root) / 2), ((-b - root) / 2)))
}

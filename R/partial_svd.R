
#'
#' Performs Lanczos bidiagonalization on a numeric n x l matrix A where \cr
#' \cr
#' AP = QB and \cr
#' A*Q = PB* + r(e_m*) \cr
#' \cr
#' where P is a n x l orthogonal matrix, \cr
#' Q is a l x m orthogonal matrix, \cr
#' B is a m x m bidiagonal matrix, \cr
#' r is the residual vector of length n, \cr
#' and e_m is a vector of length m of all 0s except the mth element.
#' 
#' @param A Numeric matrix to perform Lanczos bidiagonalization on.
#' @param p1 Initial vector of unit length
#' @param m Working subspace dimension
#' 
#' @return Returns the list:
#' \itemize{
#'   \item P - n x l orthogonal matrix.
#'   \item Q - l x m orthogonal matrix.
#'   \item B - m x m bidiagonal matrix.
#'   \item beta - the mth element in the bidiagonal of B. beta = norm(r)
#'   \item r - residual vector of length n.
#' }
#' 
#' @title Lanczos Bidiagonalization
#' @author Jesse Bannon
#' @references 
#' Augmented Implicitly Restarted Lanczos Bidiagonalization Methods, J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005.
#' 
#' @rdname pm.bidiagonalize
pm.bidiagonalize <- function(A, p1, m)
{
  l <- nrow(A)
  n <- ncol(A)
  
  P <- cbind(p1, matrix(0, n, m-1))
  Q <- cbind(A %*% p1, matrix(0, l, m-1))
  B <- vector(length = m-1) # beta - bidiagonal elements in B
  O <- vector(length = m)   # alpha - diagonal elements in B
  r <- vector(length = m)   # residuals
  
  O[1] <- pm.normalize(Q[,1])
  Q[,1] <- Q[,1] / O[1]
  
  for (j in 1:m)
  {
    r <- t(A) %*% Q[,j] - (O[j] * P[,j])
    
    # Reorthogonalize residual vector
    r <- r - (P[,1:j] %*% (t(P[,1:j]) %*% r))
    
    if (j < m)
    {
      B[j] <- pm.normalize(r)
      P[,j+1] <- r/B[j]
      
      Q[,j+1] <- A %*% P[,j+1] - (B[j] * Q[,j])
      
      # Reorthogonalize new Q vector
      Q[,j+1] <- Q[,j+1] - (Q[,1:j] %*% (t(Q[,1:j]) %*% Q[,j+1]))
      O[j+1] <- pm.normalize(Q[,j+1])
      Q[,j+1] <- Q[,j+1] / O[j+1]
    }
  }
  
  return(list(
    P = P,
    Q = Q,
    B = .biDiag(m, m, O, B),
    beta = pm.normalize(r),
    r = r
  ))
}

#' 
#' Helper function for pm.bidiagonalize. \cr
#' Returns a m x n bidiagonal matrix of the specified dimensions and the given vectors.
#' 
#' @param m Number of rows.
#' @param n Number of columns.
#' @param a Vector of length n to go on the diagonal. Defaults to all 1s.
#' @param b Vector of length n-1 to go on the bidiagonal (above the diagonal). Defaults to all 1s.
#' 
#' @return Bidiagonal matrix of the given dimensions and vectors.
#' 
#' @title Bidiagonal
#' @author Jesse Bannon
#' @rdname .biDiag
#' 
.biDiag <- function(m, n, a = rep(1, n), b = rep(1, n-1))
{
  toRet <- diag(a, m, n)
  toRet[, 2:n] <- toRet[,2:n] + diag(b, m, n-1)
  return(toRet)
}

#' 
#' Approximates partial singular values and corresponding singular vectors
#' by using implicitly restarted Lanczos bidiagonalization.
#' 
#' @param A Numeric matrix to perform partial SVD on.
#' @param nv Number of right singular vectors to estimate.
#' @param nu Number of left singular vectors to estimate.
#' @param m Working subspace dimension, larger values can speed convergence at the cost of memory.
#' @param tol Tolerance for convergence of an estimated singular value and corresponding singular vectors.
#' @param maxiter Maximum number of iterations to try for a convergence.
#' @param p1 Optional starting vector.
#' 
#' @return Returns the list:
#' \itemize{
#'   \item d - max(nv, nu) approximated singular values
#'   \item u - nu approximated left singular vectors.
#'   \item nv - nv approximated right singular vectors.
#'   \item iter - Number of iterations to converge.
#' }
#' 
#' @title Partial SVD
#' @author Jesse Bannon
#' @references 
#' Augmented Implicitly Restarted Lanczos Bidiagonalization Methods, J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005.
#' 
#' @rdname pm.partialSVD
#' @export
#' 
pm.partialSVD <- function(A, nv, nu, m = nv + 5, tol = 1e-5, maxiter = 1000, p1 = rnorm(ncol(A)))
{
  l <- nrow(A)
  n <- ncol(A)
  
  k <- max(nv, nu)
  k_orig <- k
  iter <- 1
  
  if (k / ncol(A) >= 0.5)
    warning("If you are approximating more than 50% of right singular vectors, traditional svd will work better!")
  if (!is.vector(p1) || length(p1) != n)
    p1 <- rnorm(n)
  if (m <= k)
    m <- k + 1
  if (m > min(l, n))
    m <- min(l, n) - 1
  
  p1 <- p1 / pm.normalize(p1)
  b <- pm.bidiagonalize(A, p1, m)

  while (iter < maxiter)
  {
    bSVD <- svd(b$B)
    
    if (iter == 1)
      Smax <- max(bSVD$d)
    else
      Smax <- max(Smax, max(bSVD$d[1]))
    
    residuals <- b$beta * bSVD$u[m, ]
    
    convRes <- sum(abs(residuals[1:k_orig]) < tol * Smax)
    if (convRes == k_orig)
    {
      break
    }
    else
    {
      k <- max(k, k_orig + convRes)
      if (k > m - 3)
        k <- max(m - 3, 1)
    }

    pj <- b$beta * bSVD$u[m, 1:k]
    
    b$Q <- b$Q[,1:m] %*% bSVD$u[,1:k]
    b$P <- cbind(b$P[,1:m] %*% bSVD$v[,1:k], b$r / b$beta)
    
    b$r <- (A %*% b$P[,k+1]) - rowSums(sapply(1:k, function(i) { pj[i] * b$Q[,i] }))
    b$beta <- pm.normalize(b$r)
    
    b$Q <- cbind(b$Q, b$r / b$beta)
    b$B <- cbind(diag(bSVD$d[1:k], nrow = k+1, ncol = k), c(pj, b$beta))
    
    if ((k+1) <= m) for (j in (k+1):m)
    {
      b$r <- t(A) %*% b$Q[,j] - (b$B[j,j] * b$P[,j])
      b$r <- b$r - (b$P[,1:j] %*% (t(b$P[,1:j]) %*% b$r)) # Orthogonalize with matrix P
      b$beta <- pm.normalize(b$r)
      
      if (j < m)
      {
        b$P <- cbind(b$P, b$r / b$beta)
        b$B <- cbind(b$B, c(rep(0, j-1), b$beta))
        b$Q <- cbind(b$Q, (A %*% b$P[,j+1]) - (b$beta * b$Q[,j]))

        temp <- pm.normalize(b$Q[,j+1])
        b$B <- rbind(b$B, c(rep(0, j), temp))
        b$Q[,j+1] <- b$Q[,j+1] / temp
      }
    }

    iter <- iter + 1
  }
  
  if(nu > 0) 
    u <- b$Q %*% bSVD$u[,1:nu]
  else
    u <- NULL
  
  if (nv > 0)
    v <- b$P %*% bSVD$v[,1:nv]
  else
    v <- NULL
  
  return(list(
    d = bSVD$d[1:k_orig],
    u = u,
    v = v,
    iter = iter
  ))
}


#' 
#' Computes the R matrix of a QR decomposition using the tile QR algorithm. \cr
#' The algorithm can be roughly described like this: \cr
#' \cr
#' 1. At step k the QR factorization of the diagonal tile is done. \cr
#' 2. The previous transformation modifies all other tiles along the k-th row. \cr
#' 3. Then the diagonal tile (which is now an upper triangle) is ued to annihilate all the subdiagonal tiles i one at a time. \cr
#' 4. Eack of the previous operations also modifies the tiles along rows k and i. \cr
#' \cr
#' Using the functions for each step:
#' \itemize{
#'   \item DGEQT2 (1)
#'   \item DLARFB (2)
#'   \item DTSQT2 (3)
#'   \item DSSRFB3 (4)
#' }
#' \cr
#' NOTE: This version uses lapply and assigns multiple computed tiles to A at once (tile row-wise).
#' 
#' @title pm.QR.R
#' @param A Numeric matrix.
#' @return R matrix of a QR decomposition.
#' 
#' @references
#' Parallel Tiled QR Factorization for Multicore Architectures, A. Buttari, J. Langou, J. Kurzak, J. Dongarra
#' 
#' @title Tile QR Decomposition
#' @author Jesse Bannon
#' @rdname pm.QR.R
#' @export
#'
pm.QR.R <- function(A)
{
  colsToRid <- diff16(ncol(A))
  rowsToRid <- diff16(nrow(A))
  
  A <- to16(A)
  b <- 16 # Block size
  
  colCount <- ncol(A)
  p <- as.integer(nrow(A) / b)
  q <- as.integer(ncol(A) / b)
  
  T1 <- matrix(nrow = b, ncol = b)
  
  for (k in 1:min(p,q))
  {
    kkBLK <- bl(k, k, b)
    kRows <- (k*b+1):colCount
    
    kk <- DGEQT2(A[kkBLK$r, kkBLK$c])
    T1 <- kk$T1
    A[kkBLK$r, kkBLK$c] <- kk$RV
    
    if (k+1 <= q)
    {
      A[kkBLK$r, kRows] <- do.call(cbind, lapply((k+1):q, function(n)
      {
        knBLK <- bl(k, n, b)
        DLARFB(A[knBLK$r, knBLK$c], A[kkBLK$r, kkBLK$c], T1)
      }))
    }
    
    if (k == 1) {
      print(A[1:16,])
    }
    
    if (k+1 <= p) for (n in (k+1):p)
    {
      nkBLK <- bl(n, k, b)
      nk <- DTSQT2(A[kkBLK$r, kkBLK$c], A[nkBLK$r, nkBLK$c])
      
      # Updates R section of k x k tile
      A[kkBLK$r, kkBLK$c] <- nk$RV[1:b, 1:b]
      
      # Sets tile below k x k as householder reflectors (actual reflectors have an I matrix on top of it)
      A[nkBLK$r, nkBLK$c] <- nk$RV[(b+1):(b*2), 1:b]
      T1 <- nk$T1
      
      if (k+1 <= q)
      {
        temp <- do.call(cbind, lapply((k+1):q, function(m)
        {
          kmBLK <- bl(k, m, b)
          nmBLK <- bl(n, m, b)
          DSSRFB3(A[kmBLK$r, kmBLK$c], A[nmBLK$r, nmBLK$c], A[nkBLK$r, nkBLK$c], T1)
        }))
        
        A[kkBLK$r, kRows] <- temp[1:b, ]
        A[nkBLK$r, kRows] <- temp[(b+1):(b*2), ]
      }
    }
  }
  
  A <- A[1:(nrow(A) - rowsToRid), 1:(ncol(A) - colsToRid)]
  A[lower.tri(A)] <- 0
  return(A)
}

#' 
#' Computes the R matrix of a QR decomposition using the tile QR algorithm. \cr
#' The algorithm can be roughly described like this: \cr
#' \cr
#' 1. At step k the QR factorization of the diagonal tile is done. \cr
#' 2. The previous transformation modifies all other tiles along the k-th row. \cr
#' 3. Then the diagonal tile (which is now an upper triangle) is ued to annihilate all the subdiagonal tiles i one at a time. \cr
#' 4. Eack of the previous operations also modifies the tiles along rows k and i. \cr
#' \cr
#' Using the functions for each step:
#' \itemize{
#'   \item DGEQT2 (1)
#'   \item DLARFB (2)
#'   \item DTSQT2 (3)
#'   \item DSSRFB3 (4)
#' }
#' \cr
#' NOTE: This version uses for-loops and computes individual tiles.
#' 
#' @title pm.QR.R
#' @param A Numeric matrix.
#' @return R matrix of a QR decomposition.
#' 
#' @references
#' Parallel Tiled QR Factorization for Multicore Architectures, A. Buttari, J. Langou, J. Kurzak, J. Dongarra
#' 
#' @title Tile QR Decomposition
#' @author Jesse Bannon
#' @rdname pm.QR.R
#' @export
#'
pm.QR.R2 <- function(A)
{
  n <- ncol(A)
  m <- nrow(A)
  colsToRid <- diff16(ncol(A))
  rowsToRid <- diff16(nrow(A))
  
  A <- to16(A)

  b <- 16 # Block size
  p <- as.integer(nrow(A) / b)
  q <- as.integer(ncol(A) / b)
  
  T1 <- matrix(nrow = b, ncol = b)
  
  for (k in 1:min(p,q))
  {
    kkBLK <- bl(k, k, b)
    
    kk <- DGEQT2(A[kkBLK$r, kkBLK$c])
    T1 <- kk$T1
    A[kkBLK$r, kkBLK$c] <- kk$RV
    
    if (k+1 <= q) for (n in (k+1):q)
    {
      knBLK <- bl(k, n, b)
      A[knBLK$r, knBLK$c] <- DLARFB(A[knBLK$r, knBLK$c], A[kkBLK$r, kkBLK$c], T1)
    }
    
    if (k+1 <= p) for (n in (k+1):p)
    {
      nkBLK <- bl(n, k, b)
      
      nk <- DTSQT2(A[kkBLK$r, kkBLK$c], A[nkBLK$r, nkBLK$c])
      
      # Updates R section of k x k tile
      A[kkBLK$r, kkBLK$c] <- nk$RV[1:b, 1:b]
      
      # Sets tile below k x k as householder reflectors (actual reflectors have an I matrix on top of it)
      A[nkBLK$r, nkBLK$c] <- nk$RV[(b+1):(b*2), 1:b]
      T1 <- nk$T1
      
      if (k+1 <= q) for (m in (k+1):q)
      {
        kmBLK <- bl(ibeg, m, b)
        nmBLK <- bl(n, m, b)
        
        temp <- DSSRFB3(A[kmBLK$r, kmBLK$c], A[nmBLK$r, nmBLK$c], A[nkBLK$r, nkBLK$c], T1)
        
        A[kmBLK$r, kmBLK$c] <- temp[1:b, 1:b]
        A[nmBLK$r, nmBLK$c] <- temp[(b+1):(b*2), 1:b]
      }
    }
  }
  
  #A <- A[1:(nrow(A) - rowsToRid), 1:(ncol(A) - colsToRid)]
  #A <- A[1:n, 1:n]
  #A[lower.tri(A)] <- 0
  return(A[1:m, 1:n])
}

















pm.QR.R3 <- function(A) {
  n <- ncol(A)
  m <- nrow(A)
  colsToRid <- diff16(ncol(A))
  rowsToRid <- diff16(nrow(A))
  
  A <- to16(A)
  
  b <- 16 # Block size
  p <- as.integer(nrow(A) / b)
  q <- as.integer(ncol(A) / b)
  
  T1 <- matrix(nrow = b, ncol = b)
  
  mt <- 0
  domains <- list()
  for (i in 1:4)
  {
      partitionSize <- m / 4
      mt <- partitionSize / 16
      start <- ((i - 1) * partitionSize) + 1
      end <- (i * partitionSize)
      domains[[i]] <- A[start:end, ]
  }
  
  nextMt <- mt
  proot <- 1
  
  for (k in 1:min(p,q))
  {
    
    if (k > nextMt) {
      proot <- proot + 1
      nextMt <- nextMt + mt
    }
    
    for (pi in proot:4) {
      
      ibeg <- 1
      if (pi == proot) {
        ibeg = k - ((proot - 1) * mt)
      }
      
      m <- nrow(domains[[pi]])
      p <- as.integer(nrow(domains[[pi]]) / b)

      kkBLK <- bl(ibeg, k, b)
      
      cat(sprintf("performing dgeqt2 on %d | %d,%d\n", pi, ibeg, k))
      kk <- DGEQT2(domains[[pi]][kkBLK$r, kkBLK$c])
      T1 <- kk$T1
      domains[[pi]][kkBLK$r, kkBLK$c] <- kk$RV
      
      if (k+1 <= q) for (n in (k+1):q)
      {
        knBLK <- bl(ibeg, n, b)
        cat(sprintf("performing dlarfb on %d | %d,%d\n", pi, ibeg, n))
        domains[[pi]][knBLK$r, knBLK$c] <- DLARFB(domains[[pi]][knBLK$r, knBLK$c], domains[[pi]][kkBLK$r, kkBLK$c], T1)
      }
      
      if ((ibeg + 1) <= mt) for (n in (ibeg + 1):mt)
      {
        nkBLK <- bl(n, k, b)
        print(nkBLK)
          
        nk <- DTSQT2(domains[[pi]][kkBLK$r, kkBLK$c], domains[[pi]][nkBLK$r, nkBLK$c])
        
        # Updates R section of k x k tile
        domains[[pi]][kkBLK$r, kkBLK$c] <- nk$RV[1:b, 1:b]
        
        # Sets tile below k x k as householder reflectors (actual reflectors have an I matrix on top of it)
        domains[[pi]][nkBLK$r, nkBLK$c] <- nk$RV[(b+1):(b*2), 1:b]
        T1 <- nk$T1
        
        if (k+1 <= q) for (m in (k+1):q)
        {
          kmBLK <- bl(ibeg, m, b)
          nmBLK <- bl(n, m, b)
          
          temp <- DSSRFB3(domains[[pi]][kmBLK$r, kmBLK$c], domains[[pi]][nmBLK$r, nmBLK$c], domains[[pi]][nkBLK$r, nkBLK$c], T1)
          
          domains[[pi]][kmBLK$r, kmBLK$c] <- temp[1:b, 1:b]
          domains[[pi]][nmBLK$r, nmBLK$c] <- temp[(b+1):(b*2), 1:b]
        }
      }
    }
    
    mergeEnd <- ceiling(log(4 - (proot - 1)) / log(2))
    print("------------------")
    for (mi in 1:mergeEnd) {
      p1 = proot
      p2 = p1 + 2^(mi - 1)
      
      while (p2 <= 4) {
        print("merging:")
        print(p1)
        print(p2)
        i1 <- 1
        i2 <- 1
        if (p1 == proot) {
          i1 = k - ((proot - 1) * mt)
        }
        
        i1kBLK <- bl(i1, k, b)
        i2kBLK <- bl(i2, k, b)
        
        nk <- DTTQRT(domains[[p1]][i1kBLK$r, i1kBLK$c], domains[[p2]][i2kBLK$r, i2kBLK$c])
        T1 <- nk$T1
        
        # Updates R section of k x k tile
        domains[[p1]][i1kBLK$r, i1kBLK$c] <- nk$RV[1:b, 1:b]
        # Sets tile below k x k as householder reflectors (actual reflectors have an I matrix on top of it)
        Vi2k <- nk$RV[(b+1):(b*2), 1:b]
        
        if (k+1 <= q) for (ji in (k+1):q) {
          i1jBLK <- bl(i1, ji, b)
          i2jBLK <- bl(i2, ji, b)
          
          print(Vi2k)
          temp <- DTTSSMQR(domains[[p1]][i1jBLK$r, i1jBLK$c], domains[[p2]][i2jBLK$r, i2jBLK$c], Vi2k, T1)
          domains[[p1]][i1jBLK$r, i1jBLK$c] <- temp[1:b, 1:b]
          domains[[p2]][i2jBLK$r, i2jBLK$c] <- temp[(b+1):(b*2), 1:b]
        }
        p1 <- p1 + 2^(mi)
        p2 <- p2 + 2^(mi)
      }
    }
  }
  
  m <- nrow(A)
  for (i in 1:4)
  {
    partitionSize <- m / 4
    start <- ((i - 1) * partitionSize) + 1
    end <- (i * partitionSize)
    A[start:end, ] <- domains[[i]]
  }
  
  A <- A[1:(nrow(A) - rowsToRid), 1:(ncol(A) - colsToRid)]
  A[lower.tri(A)] <- 0
  return(A)
}


#' Helper function for pm.QR.R
#' Retrieves the rows and columns for a matrix given the block row, column, and block size.
#' 
#' @title bl
#' @param p - Row number for tile matrix.
#' @param q - Column number for tile matrix.
#' @param b - Size of a tile (in elements).
#' @return r - rows of the tile. c - columns of the tile.
#' 
#' @author Jesse Bannon
#' @rdname bl
#' 
bl <- function(p, q, b)
{
  r1 <- ((p-1) * b) + 1
  c1 <- ((q-1) * b) + 1
  
  r2 <- (p * b)
  c2 <- (q * b)
  
  return(list(
    r = r1:r2,
    c = c1:c2
  ))
}

#' Helper function for pm.QR.R \cr
#' Converts a matrix of any size to a y*16 x z*16 matrix in order to have
#' 16 x 16 tiles when performing tileQR decomposition.
#' 
#' @title to16
#' @param A Numeric matrix.
#' @return y*16 x z*16 matrix containing all elements of A in the upper-left corner.
#' 
#' @author Jesse Bannon
#' @rdname to16
#' 
to16 <- function(A)
{
  m <- nrow(A)
  n <- ncol(A)
  
  colToAppend <- diff16(n)
  rowToAppend <- diff16(m)
  
  if (colToAppend != 0)
  {
    cols <- matrix(data = 0, nrow = m, ncol = colToAppend)
    A <- cbind(A, cols)
  }
  if (rowToAppend != 0)
  {
    rows <- matrix(data = 0, nrow = rowToAppend, ncol = ncol(A))
    A <- rbind(A, rows)
  }
  
  return(A)
}

#' Helper function for pm.QR.R \cr
#' Returns the difference in rows or columns of a y*16 x z*16 matrix.
#' 
#' @title diff16
#' @param a Number of rows or columns of some matrix.
#' @return Difference in rows or columns.
#' 
#' @author Jesse Bannon
#' @rdname diff16
#' 
diff16 <- function(a)
{
  dimDiff <- 16 - (a %% 16)
  if (dimDiff == 16)
    return(0)
  else
    return(dimDiff)
}

#' Helper function for pm.QR and pm.QR.R \cr
#' Returns a householder matrix for the given vector x from column i of some matrix.
#' 
#' @title Tile QR Householder Matrix
#' @param x ith column of the matrix
#' @param i Column number of the matrix
#' 
#' @author Jesse Bannon
#' @rdname .tileQRhouseholderMatrix
#' 
.tileQRhouseholderMatrix <- function(x, i)
{
  m <- length(x)
  x <- c(rep(0, i-1), x[i:m])
  
  denom <- (x[i]) + (pm.householderSign(x[i]) * pm.normalize(x))
  if (denom == 0)
  {
    v <- rep(0, m)
  }
  else
  {
    v <- x / denom
  }
  
  v[i] <- 1
  beta <- as.double(2 / (t(v) %*% v))
  if (!is.finite(beta))
  {
    beta <- 1
  }
  
  H <- diag(m) - (beta * (v %*% t(v)))
  return(list(
    H = H,
    y = v,
    beta = beta
  ))
}

#' Helper function for pm.QR.R \cr
#' Uses the name DGEQT2 based on the Fortran function used in most implementations of tileQR. \cr
#' \cr
#' Performs QR decomposition on a diagonal tile. Computes matrix R for A = QR \cr
#' and matrix T for Q = I + Y %*% T %*% t(Y) using the householder vectors.
#' 
#' @title DGEQT2
#' @param A Diagonal tile of a matrix.
#' @return R for A = QR and T for Q = I + Y %*% T %*% t(Y)
#' 
#' @author Jesse Bannon
#' @rdname DGEQT2
#' 
DGEQT2 <- function(A)
{
  n <- ncol(A)
  
  Y <- matrix(nrow = nrow(A), ncol = n)
  B <- vector(length = n)
  
  for (i in 1:n)
  {
    HouseHolder <- .tileQRhouseholderMatrix(A[,i], i)
    H <- HouseHolder$H
    Y[,i] <- HouseHolder$y
    B[i] <- HouseHolder$beta
    
    # Overwrites x with the R matrix (x = QR)
    A <- H %*% A
  }
  
  # Sets lower triangular values of R to 0 (fixes FLOP marginal errors)
  A[lower.tri(A)] <- Y[lower.tri(Y)]
  
  return(list(
    RV = A,
    T1 = YT(Y, B)
  ))
}

#' Helper function for pm.QR.R. \cr
#' Uses the name DLARFB based on the Fortran function used in most implementations of tileQR. \cr
#' \cr
#' Calculates an adjacent tile to the right of a diagonal tile using matrix multiplication. \cr
#' \cr
#' Let I be an identity matrix. \cr
#' Let V be the householder vectors stored in the diagnol tile (lowerDiagonal(VR) + I). \cr
#' Let A be the adjacent tile to the right of V. \cr
#' Let T be the orthogonal matrix calculated using the function YT \cr
#' \cr
#' A = t(I + (V %*% T %*% t(V))) %*% A
#' 
#' @title DLARFB
#' @param A An adjacent tile to the right of the diagonal block VR
#' @param VR A diagonal tile that contains the householder vectors in its lower diagonal.
#' @param T1 Matrix T calculated using the householder vectors of the diagonal tile.
#' @return Adjacent tile to replace matrix A
#' 
#' @author Jesse Bannon
#' @rdname DGEQT2
#'
DLARFB <- function(A, VR, T1)
{
  m <- nrow(VR)
  mDiag <- diag(m)
  
  VR[upper.tri(VR, diag=TRUE)] <- 0
  VR <- VR + mDiag
  
  A <- t(mDiag + (VR %*% T1 %*% t(VR))) %*% A
  return(A)
}

#' Helper function for pm.QR.R. \cr 
#' Uses the name DTSQT2 based on the Fortranfunction used in most implementations of tileQR. \cr
#' \cr
#' Calculates both a diagonal and a tile that is underneath the diagonal tile using QR decomposition. \cr
#' Let V be the upper triangular matrix of the diagonal tile VR. \cr
#' Let A be the tile underneath VR. \cr
#' [VR]      [V] \cr
#' [A ] = QR([A]) \cr
#' 
#' @title DTSQT2
#' @param VR A diagonal tile that contains an upper triangular R matrix from a QR decomposition.
#' @param A A tile underneath VR.
#' @return 2k x k matrix to overwrite VR and A.
#' 
#' @author Jesse Bannon
#' @rdname DGEQT2
#'
DTSQT2 <- function(VR, A)
{
  VR[lower.tri(VR)] <- 0
  return(DGEQT2(rbind(VR, A))) # QR
}

#' Helper function for pm.QR.R. \cr
#' Uses the name DSSRFB3 based on the Fortran function used in most implementations of tileQR. \cr
#' \cr
#' Calculates two tiles, one to the right of the diagonal matrix and the other underneath said tile
#' using matrix multiplication. \cr
#' \cr
#' Let A_kn be a tile to the right of the diagonal tile. \cr
#' Let A_mn be some tile who has the same column as A_kn but on a row beneath A_kn. \cr
#' Let V_mk be the householder vectors stored in a tile below the diagonal tile. To retrieve 
#' the full householder vectors, you must row-bind an identity matrix on top of V_mk. \cr
#' Let T_mk be the matrix T calculated using the function YT.\cr
#' Let I be a 2k x k identity matrix \cr
#' \cr
#' 
#' TODO: Latex this
#' [A_kn]                                          [A_kn] \cr
#' [A_mn] = t(I + (V_mk %*% T_mk %*% t(V_mk))) %*% [A_mn] 
#' 
#' @title DSSRFB3
#' @param A_kn Tile to the right of the diagonal tile.
#' @param A_mn Tile who has the same column as A_kn but on a row beneath A_kn.
#' @param V_mk Householder vectors stored in a tile below the diagonal tile on the same row as A_mn.
#' @param T1_mk T matrix calculated from V_mk using the function YT.
#' @return Updated A_kn and A_mn matrix row-binded together.
#' 
#' @author Jesse Bannon
#' @rdname DSSRFB3
#'
DSSRFB3 <- function(A_kn, A_mn, V_mk, T1_mk)
{
  V_mk_orig <- rbind(diag(nrow(T1_mk)), V_mk)
  return(t(diag(nrow(V_mk_orig)) + (V_mk_orig %*% T1_mk %*% t(V_mk_orig))) %*% rbind(A_kn, A_mn))
}

#' Returns a matrix W where Q = I - W %*% t(Y)
#' where Y are the householder vectors stored in a unit-triangular matrix.
#' 
#' @title WY
#' @param Y The householder vectors stored in a matrix
#' @param B Beta values of the corresponding householder vectors.
#' @return W Matrix W where Q = I - W %*% t(Y)
#' 
#' @author Jesse Bannon
#' @rdname WY
#'
WY <- function(Y, B)
{
  m <- ncol(Y)
  n <- nrow(Y)
  W <- B[1] * Y[,1]
  
  for (j in 2:m)
  {
    z <- B[j] * ((diag(n) - W %*% t(Y[,1:(j-1)])) %*% Y[,j])
    W <- cbind(W, z)
  }
  return(W)
}

#' Returns a matrix T where Q = I + Y %*% T %*% t(Y)
#' where Y are the householder vectors stored in a unit-triangular matrix
#' and T is a unit-triangular matrix.
#' 
#' @title YT
#' @param Y The householder vectors stored in a matrix
#' @param B Beta values of the corresponding householder vectors.
#' @return T Matrix T where Q = I + Y %*% T %*% t(Y)
#' 
#' @author Jesse Bannon
#' @rdname YT
#'
YT <- function(Y, B)
{
  m <- ncol(Y)
  T1 <- -B[1]
  
  for (j in 2:m)
  {
    z <- -B[j] * (T1 %*% t(Y[,1:(j-1)])) %*% Y[,j]
    
    T1 <- cbind(T1, z)
    T1 <- rbind(T1, c(rep(0, j-1), -B[j]))
  }
  return(T1)
}

#' Computes the QR decomposition of a matrix using householder vectors.
#' 
#' @title QR Decomposition
#' @param x Numeric matrix whos QR decomposition is to be computed.
#' @return List that contains the matrices Q and R.
#' 
#' @author Jesse Bannon
#' @rdname pm.QR
#' @export
pm.QR <- function(x)
{
  m <- nrow(x)
  n <- ncol(x)
  
  # Initializes Q into an m x m identity matrix
  Q <- diag(m)
  iterations <- min(n - (n == m), m)
  
  for (i in 1:iterations)
  {
    H <- QRhouseholderMatrix(x[,i], i)
    
    # Overwrites x with the R matrix (x = QR)
    x <- H %*% x
    Q <- Q %*% H
  }
  # Sets lower triangular values of R to 0 (fixes FLOP marginal errors)
  x[lower.tri(x)] <- 0
  
  if (n < m)
  {
    # Returns Q as m x n matrix opposed to its m x m form.
    Q <- Q[1:m, 1:n]
    # Returns R as n x n matrix opposed to its m x n form (bottom (m-n) rows are all 0s)
    x <- x[1:n, 1:n]
  }
  return(list(
    Q = Q,
    R = x
  ))
}


#' Returns a householder matrix for QR decomposition. \cr
#' Let m = the length of the column x. \cr
#' The m x m matrix returned is in the form of: \cr
#' \cr
#' TODO: Latex this
#' \cr
#' 1   ...  0 \cr
#' . 1        \cr
#' .   [    ] \cr
#' 0   [    ] \cr
#' \cr
#' Where the number of identity rows/columns = i-1
#' 
#' @title QRhouseholderMatrix
#' @param x Column from the R matrix while iterating in QR decomp.
#' @param i Current column in iteration
#' 
#' @author Jesse Bannon
#' 
QRhouseholderMatrix <- function(x, i)
{
  m <- length(x)
  
  # Replaces first i-1 elements with 0s
  x <- c(rep(0, i-1), x[i:m])
  
  xSign <- pm.householderSign(x[i])
  x[i] <- x[i] + (xSign * pm.normalize(x))
  beta <- 2 / sum(x^2)
  
  if (!is.finite(beta))
    beta <- 0
  
  return(diag(m) - (beta * (x %*% t(x))))
}

#' Returns the sign of the element. Different from R's version by associating 0 with positive.
#' 
#' @title Householder Sign
#' @param x Element to retrieve the sign of.
#' @return 1 if x >= 0; 0 if x < 0
#' 
#' @author Jesse Bannon
#' @rdname pm.householderSign
#' 
pm.householderSign <- function(x)
{
  if (x >= 0)
    return(1)
  else
    return(-1)
}

buildQ <- function(A)
{
  Q <- diag(nrow(A))
  for (i in 1:nrow(A))
  {
    e <- rep(0, i)
    if (i != nrow(A))
    {
      v <- c(e, A[(i + 1):nrow(A), i])
    } else {
      v <- e
    }
    
    e[i] <- 1
    beta <- 2 / sum(x^2)
    
    Q_ = diag(nrow(A)) - (beta*(v%*%t(v)))
    Q = Q_ %*% Q
  }
  return(Q)
}

DTTQRT <- function(A_kk, B_1k) {
  A_kk[lower.tri(A_kk)] <- 0
  B_1k[lower.tri(B_1k)] <- 0
  return(DGEQT2(rbind(A_kk, B_1k))) # QR
}

DTTSSMQR <- function(A_kj, B_1j, V_kj, T1_kj)
{
  V_mk_orig <- rbind(diag(nrow(T1_kj)), V_kj)
  return(t(diag(nrow(V_mk_orig)) + (V_mk_orig %*% T1_kj %*% t(V_mk_orig))) %*% rbind(A_kj, B_1j))
}

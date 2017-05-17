
#' Calculates k principle components using the distributed PCA algorithm. The algorithm can be described by: \cr \cr
#' 1. Calculate column means locally on each node (colMeans(x_i)). Send to master node to calculate column means of data matrix (colMeans(x)). \cr
#' 2. Normalize local data on each node and solve for R in QR decomposition (using tile-QR). \cr
#' 3. Pair R's from local nodes, rbind, and solve for R in QR decomposition (using tile-QR). \cr
#' 4. Repeat step 3 until you have one n x n R matrix (n being # of columns in x). \cr
#' 5. Rbind (sqrt(nrow(node#)) * (colMeans(x_i) - colMeans(x))) P times (P = # of nodes). Then Rbind n x n R matrix. Solve for V vectors in SVD. \cr
#' 6. Send first k vectors from V to each node, multiply x_i %*% V[,1:k]. Rbind results from each node in order to obtain PCs of data matrix.
#' 
#' @title pm.PCA
#' @param x Matrix to calculate principle components of.
#' @param k Number of principle components desired.
#' @param nodeInfo A 2xP matrix where P = # of nodes. [1,p]:[2,p] represent the rows contained on that node.
#' @param partialSVD logical; to approximate a SVD using implicitly restarted Lanczos bidiagonalization.
#' @param svdSubspace Applicable with partialSVD = true; Working subspace dimension of partialSVD, larger values can speed convergence at the cost of memory.
#' @param tol Applicable with partialSVD = true; Tolerance of convergence for partialSVD.
#' @param maxiter Applicable with partialSVD = true; Maximum number of iterations to converge partialSVD approximation.
#' 
#' @return k principle components.
#' 
#' @title Principle Component Analysis
#' @author Jesse Bannon
#' @rdname pm.PCA
#' @export
#' 
pm.PCA <- function(x, k = min(dim(x)), nodeInfo = matrix(c(1, nrow(x)), nrow = 2), partialSVD = FALSE, svdSubspace = k + 5, tol = 1e-05, maxiter = 1000)
{
  if (is.matrix(nodeInfo) || nrow(nodeInfo) == 2)
  {
    rows <- sort(unlist(apply(nodeInfo, 2, FUN = function(x) { x[1]:x[2] })))
    if (all(rows != 1:nrow(x)))
    {
      stop("Invalid nodeInfo. Missing rows from the data matrix x")
    }
  }
  else
  {
    stop("Invalid nodeInfo. Must be a 2xP matrix where P represents the number of nodes")
  }
  
  nodeCount <- ncol(nodeInfo)
  nodeRowCount <- apply(nodeInfo, 2, FUN = function(node) { node[2] - node[1] + 1 })
  nodeRows <- lapply(1:nodeCount, FUN = function(node) { nodeInfo[1,node]:nodeInfo[2,node] })
  dataRowCount <- length(rows)
  
  # Local colMeans on each node
  nodeColMeans <- sapply(1:nodeCount, FUN = function(node) { colMeans(x[nodeRows[[node]], ]) })
  
  # Data colMeans on the entire data matrix using local colMeans
  dataColMeans <- rowSums(sapply(1:nodeCount, FUN = function(node) { nodeRowCount[node] * nodeColMeans[,node] })) / dataRowCount
  
  # Performs QR decomposition on every node
  nodeQR <- lapply(1:nodeCount, function(nodeNumber) { pm.QR.R(t(apply(x[nodeRows[[nodeNumber]], ], 1, nodeColMeans[,nodeNumber], FUN = "-"))) })
  
  # Pairs and rbinds R matrices to solve for a larger R until there is one.
  i <- nodeCount
  while(length(nodeQR) > 1)
  {
    j <- ceiling(i / 2)
    i <- floor(i / 2)
    nodeQR <- lapply(1:i, function(node, inc) { pm.QR.R(rbind(nodeQR[[node]], nodeQR[[node + inc]])) }, j)
  }
  
  # Row binds the means and the master R matrix
  nodeQR <- rbind(t(sapply(1:nodeCount, function(node) { sqrt(nodeRowCount[node]) * (nodeColMeans[,node] - dataColMeans) })), nodeQR[[1]])
  
  # Calculates right singular vectors by solving SVD on qr(master R matrix)
  print(nodeQR)
  if (partialSVD)
  {
    singularVecs <- pm.partialSVD(pm.QR.R(nodeQR), nv = k, nu = 0, m = svdSubspace, tol = tol, maxiter = maxiter)$v
  }
  else
  {
    singularVecs <- svd(pm.QR.R(nodeQR))$v[,1:k]
  }
  
  print(singularVecs)
  
  # Multiplies the singularVecs on each local node, then sends them to the master node to rbind. Must be done in order of matrix rows
  pc <- do.call(rbind, sapply(order(nodeInfo[1,]), function(node) { t(apply(x[nodeRows[[node]], ], 1, dataColMeans, FUN = "-")) %*% singularVecs }, simplify = FALSE))
  
  return(pc)
}

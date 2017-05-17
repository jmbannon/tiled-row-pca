
#' Returns the vector norm of the given vector or the columns of a given matrix.
#' 
#' @title Vector Normalize
#' @param x Vector or matrix to calculate the vector norm(s) of.
#' @return Normalized vector of x
#' 
#' @author Jesse Bannon
#' @rdname pm.normalize
#' 
pm.normalize <- function(x)
{
  if (is.vector(x))
  {
    return(sqrt(sum(x^2)))
  }
  else if (is.matrix(x))
  {
    return(apply(x, 2, function(xi) { sqrt(sum(xi^2)) }))
  }
  else
  {
    stop("Must supply a vector or matrix.")
  }
}

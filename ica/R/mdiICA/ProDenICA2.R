ProDenICA2<-function (x, k = p, W0 = NULL, whiten = FALSE, maxit = 20, thresh = 1e-07, 
          restarts = 0, trace = FALSE, Gfunc = GPois, eps.rank = 1e-07, 
          ...) 
{ print('dim(x)')
  print(dim(x))
  this.call = match.call()
  p <- ncol(x)
  n <- nrow(x)
  x <- scale(x, T, F)
  if (whiten) {
    sx <- svd(x)
    condnum = sx$d
    condnum = condnum/condnum[1]
    good = condnum > eps.rank
    rank = sum(good)
    if (k > rank) {
      warning(paste("Rank of x is ", rank, "; k reduced from", 
                    k, " to ", rank, sep = ""))
      k = rank
    }
    x <- sqrt(n) * sx$u[, good]
    whitener = sqrt(n) * scale(sx$v[, good], FALSE, sx$d[good])
  }
  else whitener = NULL
  if (is.null(W0)) 
    W0 <- matrix(rnorm(p * k), p, k)
  else k = ncol(W0)
  W0 <- ICAorthW(W0)
  GS <- matrix(0, n, k)
  gS <- GS
  gpS <- GS
  s <- x %*% W0
  flist <- as.list(1:k)
  print('dim(s)')
  print(dim(s))
  print(dim(s[,1]))
  for (j in 1:k) flist[[j]] <- Gfunc(s[, j], ...)
  
  flist0 <- flist
  crit0 <- mean(sapply(flist0, "[[", "Gs"))
  while (restarts) {
    W1 <- matrix(rnorm(p * k), p, k)
    W1 <- ICAorthW(W1)
    s <- x %*% W1
    print(paste('dim(s) ',dim(s[,1])))
    for (j in 1:k) flist[[j]] <- Gfunc(s[, j], ...)
    crit <- mean(sapply(flist, "[[", "Gs"))
    if (trace) 
      cat("old crit", crit0, "new crit", crit, "\n")
    if (crit > crit0) {
      crit0 <- crit
      W0 <- W1
      flist0 <- flist
    }
    restarts <- restarts - 1
  }
  nit <- 0
  nw <- 10
  repeat {
    nit <- nit + 1
    gS <- sapply(flist0, "[[", "gs")
    #print('dim(gS) ',dim(gS))
    
    gpS <- sapply(flist0, "[[", "gps")
    #print('dim(gpS) ',dim(gpS))
    t1 <- t(x) %*% gS/n
    t2 <- apply(gpS, 2, mean)
    W1 <- t1 - scale(W0, F, 1/t2)
    print(paste('W ',W1))
    W1 <- ICAorthW(W1)
    
    nw <- amari(W0, W1)
    if (trace) 
      cat("Iter", nit, "G", crit0, "crit", nw, "\n")
    W0 <- W1
    if ((nit > maxit) | (nw < thresh)) 
      break
    s <- x %*% W0
    for (j in 1:k) flist0[[j]] <- Gfunc(s[, j], ...)
    crit0 <- mean(sapply(flist0, "[[", "Gs"))
  }
  rl = list(W = W0, negentropy = crit0, s = x %*% W0, whitener = whitener, 
            call = this.call)
  rl$density = lapply(flist0, "[[", "density")
  class(rl) = "ProDenICA"
  rl
}
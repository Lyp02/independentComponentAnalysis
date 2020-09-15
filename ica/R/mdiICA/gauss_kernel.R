gauss_kenrel<-function(x,knots,bw=0.1){
  n<-length(x)
  m<-length(knots)
  gram   <-array(0.0,dim=c(n,m))
  gram1d <-array(0.0,dim=c(n,m))
  gram2d <-array(0.0,dim=c(n,m))
  
  distance <-array(0.0,dim=c(n,m))
  for (i in 1:n){
    for (j in 1:m){
      distance[i,j]<-x[i]-knots[j]
      gram[i,j]<-exp(-1.0*bw*distance[i,j]^2)
      gram1d[i,j]<-(-2.0*bw)*distance[i,j]*gram[i,j]
      gram2d[i,j]<-gram[i,j]*(4*(bw*distance[i,j])^2-2*bw)
    }
  }
  r1 =list(g0=gram, g1 = gram1d, g2 = gram2d)
  r1
}
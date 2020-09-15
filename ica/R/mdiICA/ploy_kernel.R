poly_kernel<-function(x,knots,s=1.0,d=3){
  n<-length(x)
  m<-length(knots)
  gram   <-array(0.0,dim=c(n,m))
  gram1d <-array(0.0,dim=c(n,m))
  gram2d <-array(0.0,dim=c(n,m))
  
  distance <-array(0.0,dim=c(n,m))
  for (i in 1:n){
    for (j in 1:m){

      gram[i,j]<-(s*x[i]*knots[j]+1)^(d)
      gram1d[i,j]<-s*knots[j]*d*(s*x[i]*knots[j]+1)^(d-1)
      gram2d[i,j]<-((s*knots[j])^(2))*(d)*(d-1)*(s*x[i]*knots[j]+1)^(d-2)
    }
  }
  r1 =list(g0=gram, g1 = gram1d, g2 = gram2d)
  r1
}
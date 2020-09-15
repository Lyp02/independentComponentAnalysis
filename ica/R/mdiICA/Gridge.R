Gridge<-function (x,tol=1e-6, df = 2, B = 500, order = 1, widen = 1.2, density.return = FALSE, 
          ...) 
{
  print("Gridge begin")
  x <- drop(scale(x))
  xs <-array(0,dim=c(length(x),5))
  xs[,1]<-1.0
  xs[,2]<-x
  xs[,3]<-(x^2)
  xs[,4]<-(x^3)
  xs[,5]<-(x^4)
  
  xs1 <-array(0,dim=c(length(x),5))
  xs1[,1]<-0.0
  xs1[,2]<-1.0
  xs1[,3]<-(2*x)
  xs1[,4]<-(3*x^2)
  xs1[,5]<-(4*x^3)
  
  xs2 <-array(0,dim=c(length(x),5))
  xs2[,1]<-0.0
  xs2[,2]<-0.0
  xs2[,3]<-(2.0)
  xs2[,4]<-(6*x)
  xs2[,5]<-(12*x^2)
  
  n <- length(x)
  rangex <- range(x)
  if (order == 1) 
    rx <- rangex
  else {
    rx <- sort(x)[c(order, n - order + 1)]
  }
  rx <- ylim.scale(rx, diff(rx) * widen)
  xg <- seq(from = rx[1], to = rx[2], length = B)
  gaps <- diff(rx)/(B - 1)
  xcuts <- c(min(rangex[1], rx[1]) - gaps/2, xg[-B] + gaps/2, 
             max(rangex[2], rx[2]) + gaps/2)
  ys <- as.vector(table(cut(x, xcuts)))
  gxg <- dnorm(xg)
  gs1 <-array(0,dim=c(length(xg)))
  gs2 <-array(1,dim=c(length(xg)))
  lambda_seqs <-10^seq(4,-4,length.out = 1000)
  xgs <-array(0,dim=c(length(xg),5))
  xgs[,1]<-1.0
  xgs[,2]<-xg
  xgs[,3]<-(xg^2)
  xgs[,4]<-(xg^3)
  xgs[,5]<-(xg^4)
  it =-1
  xgs <-as.matrix(xgs)
  while(sum((gs1-gs2)^2)>tol){
    it =it +1
    print('it')
    print(it)
    gs2 <-gs1
    mu <-gxg*exp(gs1)
    ws <-mu
    zs <-(gs1 +(ys-mu)/mu)
    fit1 <-glmnet(xgs,zs,weights = ws,alpha=0,lambda=lambda_seqs,intercept=FALSE,family = "gaussian")
    print("$$")
    l =1
    h =1000
    mid =as.integer((l+h)/2)
    while(l<h & abs(fit1$df[mid]-df)>tol){
      if(fit1$df[mid]>df){h =mid-1}
      else{l =mid+1}
      mid =as.integer((l+h)/2)
    }
    print(paste("df max ",max(fit1$df)))
    print(paste("df min ",min(fit1$df)))
    print("##")
    print("mid")
    print(mid)
    print("df")
    print(fit1$df[mid])
    gs1 <-predict(fit1,newx = xgs,s=fit1$lambda[mid])
  }
  Gs <-gs1
  
  if (density.return) {
    density = list(x = xg, y = exp(Gs + logb(gxg)))
  }
  beta =as.vector(fit1$beta[mid])
  Gs <-xs%*%beta
  gs <-xs1%*%beta
  gps<-xs2%*%beta
  print("&&")

  rl = list(Gs = Gs, gs = gs, gps = gps)
  if (density.return) 
    rl$density = density
  rl
  print("Gridge end")
}
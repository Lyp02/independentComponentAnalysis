Gmle2<-function (x, df = 6, B = 500, order = 1, widen = 1.2, density.return = FALSE, lbd = 1,nu=6, maxiters=100,op="cvxr",method="ls",
                ...) 
{
  print(paste("Gmle ","begin"))
  x <- drop(scale(x))
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
  
  ys <-array(ys,dim=c(B,1))/N
  
  bw <-1.0/(2*nu*nu*gaps*gaps)
  
  
  
  w <-array(1.0,dim=c(B,1))
  
  ALPHA=0.01
  BETA =0.5
  NTTOL =1e-8
  MU =20
  TOL =1e-5
  t1 =1
  #K<-gauss_kenrel(x=xg,knots=xg,bw=bw)$g0
  
  #K<-poly_kernel(x=xg,knots=xg,s=1.0,d=3)$g0
  R<-array(0,dim=c(B-2,B-2))
  Q<-array(0,dim=c(B,B-2))
  for(j in 2:B-1){
    Q[j-1,j-1]<-1.0/gaps
    Q[j,j-1]<--2.0/gaps
    Q[j+1,j-1]<-1.0/gaps
    R[j-1,j-1]<-2*gaps/3
    if(j<B-1){
      R[j-1,j]<-gaps/6
      R[j,j-1]<-gaps/6
    }
  }
  print(paste("R rank ",qr(R)$rank))
  RinQt <-solve(R,t(Q))
  K <-Q%*%RinQt
  print(paste("K rank ",qr(K)$rank))
  if(op=="cvxr"){
    v<-Variable(B,1)
    obj<--1.0*sum(ys*log(K%*%v))+lbd*quad_form(v,K)
    constr<-list(K%*%v>=1e-9,sum(K%*%v)==1.0)
    prob<-Problem(Minimize(obj),constr)
    result <-solve(prob)
    val<-result$value
    w  <-result$getValue(v)
    print(paste('w',w))
    print(paste("loss ",val))
  }else{
    while(maxiters>0){
      
      ps <-K%*%w
      print(paste('min(ps) ',min(ps)))
      ps[ps<1e-9]<-1e-9
      print(paste('min(ps) ',min(ps)))
      val  <-t1*sum(ps*gaps-ys*log(ps))-sum(log(ps))+t1*lbd*t(w)%*%K%*%w
      print(paste('val ',val))
      grad <-t1*apply((array((gaps-ys/ps),dim=c(B,B)))*K, 2, sum)-apply((array(1.0/ps,dim=c(B,B)))*K, 2, sum)+t1*2*lbd*K%*%w
      hess <-t1*(t(K)%*%diag((ys/(ps*ps))[,1])%*%K)+t(K)%*%diag((1.0/(ps*ps))[,1])%*%K+t1*2*lbd*K
      print('1')
      print(paste('grad ',grad))
      
      
      e<-0.000005
      r<-1.1
      
      h<-hess
      while(qr(h)$rank<B){
        h<-hess+e*diag(B)
        e<-r*e
      }
      hess<-h
      
      
      
      step <--1.0*solve(hess,grad)
      print('step ',step)
      print('2')
      fprime<-sum(grad*step)
      print(paste('fprime ',fprime))
      if(abs(fprime)<NTTOL){
        gap <-(B/t1)
        if(gap<TOL)
          break
        else
          t1<-MU*t1
      }else{
        tls=1.0
        new_w<-w+tls*step

        while(min(K%*%new_w)<=0){
          print(paste('min(ps) ',min(K%*%new_w)))
          tls<-BETA*tls
          new_w<-w+tls*step
        }
        print('3')
        new_ps <-K%*%new_w
        new_val  <-t1*sum(new_ps*gaps-ys*log(new_ps))-sum(log(new_ps))+t1*lbd*t(new_w)%*%K%*%new_w
        while(new_val>=(val+tls*ALPHA*fprime)){
          tls<-BETA*tls
          new_w<-w+tls*step
          new_ps <-K%*%new_w
          new_val  <-t1*sum(new_ps*gaps-ys*log(new_ps))-sum(log(new_ps))+t1*lbd*t(new_w)%*%K%*%new_w
        }
        w<-w+tls*step
        
        print(paste("loss ",val))
        
      }
      maxiters<-maxiters-1
    }
  }
  
  
  
  
  
  if (density.return) {
    density = list(x = xg, y =exp(K%*%w)[,1] )
  }
  gk <-gauss_kenrel(x=x,knots = xg,bw=bw)
  gk <-poly_kernel(x=xg,knots=xg,s=1.0,d=3)
  p0 <-(gk$g0)%*%w
  p1 <-(gk$g1)%*%w
  p2 <-(gk$g2)%*%w
  
  Gs <-p0[,1]
  gs <-p1[,1]
  gps<-p2[,1]
  
  
  
  rl = list(Gs = Gs, gs = gs, gps = gps)
  if (density.return) 
    rl$density = density
  rl
}
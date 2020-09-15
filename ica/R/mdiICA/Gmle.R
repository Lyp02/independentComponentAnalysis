#difficult to pick the inverse Kernel
Gmle<-function (x, df = 6, B = 500, order = 1, widen = 1.2, density.return = FALSE, lbd = 0.001,nu=0.5, maxiters=100,op="cvxr",method="ls",
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
  #bw<-0.00002

  
  
  w <-array(1.0,dim=c(B,1))
  
  ALPHA=0.01
  BETA =0.5
  NTTOL =1e-8
  MU =20
  TOL =1e-5
  t1 =1
  
  K<-gauss_kenrel(x=xg,knots=xg,bw=bw)$g0

  
  #K<-poly_kernel(x=xg,knots=xg,s=1.0,d=3)$g0
  if(op=="cvxr"){
    h<-K
    eig_dec<-eigen(h,symmetric = TRUE)
    eig_vals<-eig_dec$values
    eig_vecs<-eig_dec$vectors
    if(min(eig_vals)<=0)
      eig_vals<-eig_vals-min(eig_vals)+1e-7
      print(paste('eig_val min ',min(eig_vals)))
    h<-eig_vecs%*%diag(eig_vals)%*%t(eig_vecs)
    K<-h
    print(paste(' k rank ',qr(K)$rank))
    
    v<-Variable(B,1)
    obj<-sum(gaps*K%*%v)-sum(ys*log(K%*%v))+lbd*quad_form(v,K)
    constr<-list(K%*%v >=1e-7)
    prob<-Problem(Minimize(obj),constr)
    result <-solve(prob)
    val<-result$value
    w  <-result$getValue(v)
    print(paste("loss ",val))
  }else{
    while(maxiters>0){
      
      ps <-K%*%w
      val  <-t1*sum(ps*gaps-ys*log(ps))-sum(log(ps))+t1*lbd*t(w)%*%K%*%w
      grad <-t1*apply((array((gaps-ys/ps),dim=c(B,B)))*K, 2, sum)-apply((array(1.0/ps,dim=c(B,B)))*K, 2, sum)+t1*2*lbd*K%*%w
      hess <-t1*(t(K)%*%diag((ys/(ps*ps))[,1])%*%K)+t(K)%*%diag((1.0/(ps*ps))[,1])%*%K+t1*2*lbd*K
      e<-0.005
      r<-1.1
      
      h<-hess
      eig_dec<-eigen(h,symmetric = TRUE)
      eig_vals<-eig_dec$values
      eig_vecs<-eig_dec$vectors
      if(min(eig_vals)<=0)
        eig_vals<-eig_vals-min(eig_vals)+1e-7
      h<-eig_vecs%*%diag(eig_vals)%*%t(eig_vecs)
      hess<-h
      
      step <--1.0*solve(hess,grad)
      
      
      fprime<-sum(grad*step)
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
          tls<-BETA*tls
          new_w<-w+tls*step
        }
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
    density = list(x = xg, y =(K%*%w)[,1] )
  }
  gk <-gauss_kenrel(x=x,knots = xg,bw=bw)
  #gk <-poly_kernel(x=xg,knots=xg,s=1.0,d=3)
  p0 <-(gk$g0)%*%w
  p1 <-(gk$g1)%*%w
  p2 <-(gk$g2)%*%w
  
  Gs <-(log(p0))[,1]
  gs <-((p1/p0))[,1]
  gps<-((p2*p0-p1*p1)/(p0*p0))[,1]
  
  
  
  rl = list(Gs = Gs, gs = gs, gps = gps)
  if (density.return) 
    rl$density = density
  rl
}
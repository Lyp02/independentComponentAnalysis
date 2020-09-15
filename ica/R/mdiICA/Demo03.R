thetas<-seq(from=0,to=pi,length.out=30)
w<-array(0,dim=c(2,1))
contrast<-array(0,dim=c(2,4,30))
contrast2<-array(0,dim=c(2,4,30))
choices<-c("c","m")
targets<-array(0,dim=c(2,2,2))
p=2
angles =array(0,dim=c(2,2))
for (i in 1:2){
  dist <-choices[i]
  A0<-mixmat(p)
  s<-scale(cbind(rjordan(dist,N),rjordan(dist,N)))
  
  A0[1,1]<-cos(pi/4)
  A0[2,1]<-sin(pi/4)
  A0[1,2]<-(-1.0*sin(pi/4))
  A0[2,2]<-cos(pi/4)
  print("A0")
  print(A0)
  x <- s %*% A0
  ###Whiten the data
  x <- scale(x, TRUE, FALSE)
  sx <- svd(x)	### orthogonalization function
  x <- sqrt(N) * sx$u
  target <- solve(A0)
  target <- diag(sx$d) %*% t(sx$v) %*% target/sqrt(N)
  targets[i,,]<-target
  angles[i,1]-atan(abs(target[2,1]/target[1,1]))
  angles[i,2]<-atan(abs(target[2,2]/target[1,2]))
  for (j in 1:30){
    w[1,1]<-cos(thetas[j])
    w[2,1]<-sin(thetas[j])
    s<-x%*%w
    xnorm<-rnorm(1000000)
    contrast[i,1,j]<-(mean(G0(s)$Gs)-mean(G0(xnorm)$Gs))^2 
    contrast[i,2,j]<-(mean(G1(s)$Gs)-mean(G1(xnorm)$Gs))^2
    contrast[i,3,j]<-mean(GPois(s)$Gs)
    contrast[i,4,j]<-(mean((Gmdi(s)$Gs)))
    
    contrast2[i,1,j]<-(mean(G0(s)$Gs)-mean(G0(xnorm)$Gs))^2 
    contrast2[i,2,j]<-(mean(G1(s)$Gs)-mean(G1(xnorm)$Gs))^2
    contrast2[i,3,j]<-mean(GPois2(s)$zs)
    contrast2[i,4,j]<-(mean((Gmdi2(s)$zs)))
  }
  contrast[i,1,]<-(contrast[i,1,]/max(contrast[i,1,])) 
  contrast[i,2,]<-(contrast[i,2,]/max(contrast[i,2,]))
  contrast[i,3,]<-(contrast[i,3,]/max(contrast[i,3,]))
  contrast[i,4,]<-(contrast[i,4,]/max(contrast[i,4,]))
  
  contrast2[i,1,]<-(contrast2[i,1,]/max(contrast2[i,1,])) 
  contrast2[i,2,]<-(contrast2[i,2,]/max(contrast2[i,2,]))
  contrast2[i,3,]<-(contrast2[i,3,]/max(contrast2[i,3,]))
  contrast2[i,4,]<-(contrast2[i,4,]/max(contrast2[i,4,]))
  
}

pdf("methodsCompareUniforms.pdf")
plot(thetas,contrast[1,1,],type="b",pch=15,lty=1,lwd=1,col="red",main="Sources:Uniforms",
     xlab=expression(theta),ylab="Index",ylim=c(0,1.0))
lines(thetas,contrast[1,2,],type="b",pch=16,lty=1,lwd=1,col="orange")
lines(thetas,contrast[1,3,],type="b",pch=17,lty=1,lwd=1,col="blue")
lines(thetas,contrast[1,4,],type="b",pch=18,lty=1,lwd=1,col="green")
abline(v=c(pi/4,3*pi/4),lwd=1,lty=1,col="gray")
legend("bottomright", inset=.05,pch=c(15,16,17,18),lty=c(1,1,1,1),lwd=c(1,1,1,1),legend=c("FastICA G1","FastICA G2","ProDenICA","proposed algorithm"),
       col=c("red","orange","blue","green"))
dev.off()
pdf("methodsCompareGaussMix.pdf")
plot(thetas,contrast[2,1,],type="b",pch=15,lty=1,lwd=1,col="red",main="Sources:Gaussian Mixtures",
     xlab=expression(theta),ylab="Index",ylim=c(0,1.0))
lines(thetas,contrast[2,2,],type="b",pch=16,lty=1,lwd=1,col="orange")
lines(thetas,contrast[2,3,],type="b",pch=17,lty=1,lwd=1,col="blue")
lines(thetas,contrast[2,4,],type="b",pch=18,lty=1,lwd=1,col="green")
abline(v=c(pi/4,3*pi/4),lwd=1,lty=1,col="gray")
legend("bottomright", inset=.05,lty=c(1,1,1,1),pch=c(15,16,17,18),lwd=c(1,1,1,1),legend=c("FastICA G1","FastICA G2","ProDenICA","proposed algorithm"),
       col=c("red","orange","blue","green"))
dev.off()

pdf("methodsCompareUniforms2.pdf")
plot(thetas,contrast[1,3,],type="b",pch=15,lty=1,lwd=1,col="blue",main="Sources:Uniforms",
     xlab=expression(theta),ylab="Index")
lines(thetas,contrast2[1,3,],type="b",pch=16,lty=1,lwd=1,col="green")
abline(v=c(pi/4,3*pi/4),lwd=1,lty=1,col="gray")
legend("bottomright", inset=.05,pch=c(15,16),lty=c(1,1),lwd=c(1,1),legend=c("proposed algorithm g","proposed algorithm z"),
       col=c("blue","green"))
dev.off()
pdf("methodsCompareGaussMix2.pdf")
plot(thetas,contrast[2,3,],type="b",pch=15,lty=1,lwd=1,col="blue",main="Sources:Gaussian Mixtures",
     xlab=expression(theta),ylab="Index")
lines(thetas,contrast2[2,3,],type="b",pch=16,lty=1,lwd=1,col="green")
abline(v=c(pi/4,3*pi/4),lwd=1,lty=1,col="gray")
legend("bottomright", inset=.05,pch=c(15,16),lty=c(1,1),lwd=c(1,1),legend=c("proposed algorithm g","proposed algorithm z"),
       col=c("blue","green"))
dev.off()
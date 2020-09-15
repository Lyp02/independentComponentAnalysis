G2<-function (x, df = 6, B = 500, order = 1, widen = 1.2, density.return = FALSE, 
              ...) 
{
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
  
  gxg <- dnorm(xg)
  weights<-0.5*gaps*gxg
  zs<-((ys-gaps*gxg)/(gaps*gxg))
  
  
  mdi.fit <- smooth.spline(xg, zs, weights, df)
  Gs <- predict(mdi.fit, x, deriv = 0)$y
  gs <- predict(mdi.fit, x, deriv = 1)$y
  gps <- predict(mdi.fit, x, deriv = 2)$y
  
  
  
  if (density.return) {
    Gs2 <- predict(mdi.fit, xg, deriv = 0)$y
    density = list(x = xg, y = exp(Gs2 + logb(gxg)))
  }
  
  
  
  
  rl = list(Gs = Gs, gs = gs, gps = gps)
  if (density.return) 
    rl$density = density
  rl
}
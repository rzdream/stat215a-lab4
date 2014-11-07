mixGaussian <- function (sampledata, nclusters, iter.max){
  
  library("mvtnorm")
  library("DirichletReg")
  library("matrixStats")
  library("mnormt")
#  library("doParallel")
#  registerDoParallel(4)
  sampledata=as.matrix(sampledata)
  dimData=dim(sampledata)
  TT=dimData[1]

#check if the sampledata generated properly
#pinum=rep(0,2)
#for (t in 1:TT){
#  pinum[xhid[t]]=pinum[xhid[t]]+1
#}
#pi=pinum/TT


# E-M algorithm
  pi=rdirichlet(1,rep(1/nclusters,nclusters))
  sampleMean=colMeans(sampledata)
  mu=matrix(seq(1/nclusters, 1, 1/nclusters),nclusters,1)%*%sampleMean
  sigma=array(1, dim=c(nclusters, dimData[2], dimData[2]))
  for (i in 1:nclusters){
    sigma[i,,] = sigma[i,,]+diag(rep(1,dimData[2]))
  }
  it=0

while (it < iter.max){
  # E step
  gamma=matrix(1/2,TT,nclusters)
  for (t in 1:TT){
    dtemp=rep(1,nclusters)
    for (i in 1:nclusters){
      dtemp[i]=dmnorm(sampledata[t,], mean=mu[i,], varcov=sigma[i,,])
    }
    gamma[t,]=pi*dtemp/sum(pi*dtemp)
  }
  # M step
  for (i in 1:nclusters){
    mu[i,]=colSums(gamma[,i]*sampledata)/sum(gamma[,i])
  }
 
  for (i in 1:nclusters){
    s=matrix(0,dimData[2],dimData[2])
    for (t in 1:TT){
      s=s+gamma[t,i]*(sampledata[t,]-mu[i,])%*%t(sampledata[t,]-mu[i,])
    }
    sigma[i,,]=s/sum(gamma[,i])
  }
  
  pi=colSums(gamma)/TT              
  
  it=it+1
  print(it)
  print(pi)
  print(mu)
#  print(sigma)
}
label=rep(1,TT)
for (t in 1:TT){
  dtemp=rep(1,nclusters)
  for (i in 1:nclusters){
    dtemp[i]=dmnorm(sampledata[t,], mean=mu[i,], varcov=sigma[i,,])
  }
  label[t]=which.max(dtemp)
}
result = list(mu, label)
names(result) = c('centers','label')
return (result)
}

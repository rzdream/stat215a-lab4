### comparison of different models with different training/testing sets

library(dplyr)
library(ggplot2)
library(matrixStats)
library(MASS)
library(randomForest)
library(doParallel)
library(caret)
registerDoParallel(10)
#setwd("/Users/rzdream/Dropbox/2014Fall/STAT215/lab4/image_data")
setwd("/accounts/class/s215a/s215a-9/lab4")
# Get the data for three images

image1 <- read.table('image1.txt', header=F)
image2 <- read.table('image2.txt', header=F)
image3 <- read.table('image3.txt', header=F)

# Add informative column names.
collabs <- c('y','x','label','NDAI','SD','CORR','DF','CF','BF','AF','AN')
names(image1) <- collabs
names(image2) <- collabs
names(image3) <- collabs

# Standardize the data
imageS1=image1
imageS2=image2
imageS3=image3
imageS1[,4:11] = image1[,4:11]/(matrix(1,dim(image1)[1],1)%*%colSds(as.matrix(image1[,4:11])))
imageS2[,4:11] = image2[,4:11]/(matrix(1,dim(image2)[1],1)%*%colSds(as.matrix(image2[,4:11])))
imageS3[,4:11] = image3[,4:11]/(matrix(1,dim(image3)[1],1)%*%colSds(as.matrix(image3[,4:11])))

lrmData=image1[image1$label != 0,]
lrmDataS=imageS1[imageS1$label != 0,]
idx_train=sample(c(T,F), size = nrow(lrmData), 
                 prob = c(.8,.2), replace = T)
trainData=lrmData[idx_train,]
trainDataS=lrmDataS[idx_train,]
testData1=lrmData[!idx_train,]
testData2=image2[image2$label!=0,]
testData3=image3[image3$label!=0,]
testDataS1=lrmDataS[!idx_train,]
testDataS2=imageS2[image2$label!=0,]
testDataS3=imageS3[image3$label!=0,]

trainData=list(trainData, trainDataS)
testData=list(testData1, testData2, testData3, testDataS1, testDataS2, testDataS3)
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

AccRate <- function(trainData, testData, method){
  if (method=='glm'){
    lrmResult=glm(as.factor(label) ~ NDAI+SD+CORR+DF+AF+BF+CF+AN, family=binomial(link='logit'), 
                  data = trainData)
    pred_lrm <- predict(lrmResult,testData,type='response')
    pred_lrm_label=pred_lrm
    pred_lrm_label[which(pred_lrm_label>0.5)]=1
    pred_lrm_label[which(pred_lrm_label<0.5)]=-1
    # acc is the accurate rate
    acc=length(which(pred_lrm_label==testData$label))/dim(testData)[1]
    return (acc)
  }
  if (method=='lda'){
    ldaResult = lda(label~ NDAI+SD+CORR+DF, data = trainData)
    pred_lda = predict(ldaResult,testData)
    prob_lda=pred_lda$posterior[,2]
    pred_lda_label=prob_lda
    pred_lda_label[which(pred_lda_label>0.5)]=1
    pred_lda_label[which(pred_lda_label<0.5)]=-1
    acc=length(which(pred_lda_label==testData$label))/dim(testData)[1]
    return (acc)    
  }
  if (method=='qda'){
    qdaResult = qda(label~ NDAI+SD+CORR+DF, data = trainData)
    pred_qda = predict(qdaResult,testData)
    prob_qda=pred_qda$posterior[,2]
    pred_qda_label=prob_qda
    pred_qda_label[which(pred_qda_label>0.5)]=1
    pred_qda_label[which(pred_qda_label<0.5)]=-1
    acc=length(which(pred_qda_label==testData$label))/dim(testData)[1]
    return (acc)    
  }
  if (method=='rf'){
    rfResult = randomForest(data=trainData, as.factor(label)~NDAI+CORR+
                              SD+DF,importance=TRUE, ntree=100)    
    pred_rf = predict(rfResult,testData)
    pred_rf_label=pred_rf
    acc=length(which(pred_rf_label==testData$label))/dim(testData)[1]
    return (acc)
  }
  if (method=='qda_kmeans'){
    qdaResult = qda(label~ NDAI+SD+CORR+DF, data = trainData)
    kmeansResult=kmeans(testData[,c(4,5,6,7)], centers=2, iter.max=100)
    test_label=rep(1,dim(testData)[1])
    dis11=sum((kmeansResult$centers[1,]-qdaResult$means[1,])^2)
    dis12=sum((kmeansResult$centers[1,]-qdaResult$means[2,])^2)
    dis21=sum((kmeansResult$centers[2,]-qdaResult$means[1,])^2)
    dis22=sum((kmeansResult$centers[2,]-qdaResult$means[2,])^2)
    if (dis11+dis22 < dis12+dis21){
      test_label[which(kmeansResult$cluster==1)]=-1
    } else {
      test_label[which(kmeansResult$cluster==2)]=-1
    }
    acc=length(which(test_label==testData$label))/dim(testData)[1]
    return (acc)
  }
  if (method=='lda_kmeans'){
    ldaResult = lda(label~ NDAI+SD+CORR+DF, data = trainData)
    kmeansResult=kmeans(testData[,c(4,5,6,7)], centers=2, iter.max=100)
    test_label=rep(1,dim(testData)[1])
    dis11=sum((kmeansResult$centers[1,]-ldaResult$means[1,])^2)
    dis12=sum((kmeansResult$centers[1,]-ldaResult$means[2,])^2)
    dis21=sum((kmeansResult$centers[2,]-ldaResult$means[1,])^2)
    dis22=sum((kmeansResult$centers[2,]-ldaResult$means[2,])^2)
    if (dis11+dis22 < dis12+dis21){
      test_label[which(kmeansResult$cluster==1)]=-1
    } else {
      test_label[which(kmeansResult$cluster==2)]=-1
    }
    acc=length(which(test_label==testData$label))/dim(testData)[1]
    return (acc)
  }
  if (method=='qda_mg'){
    qdaResult = qda(label~ NDAI+SD+CORR+DF, data = trainData)
    mgemResult=mixGaussian(testData[,c(4,5,6,7)], 2, 20)   
    test_label=rep(1,dim(testData)[1])
    dis11=sum((mgemResult$centers[1,]-qdaResult$means[1,])^2)
    dis12=sum((mgemResult$centers[1,]-qdaResult$means[2,])^2)
    dis21=sum((mgemResult$centers[2,]-qdaResult$means[1,])^2)
    dis22=sum((mgemResult$centers[2,]-qdaResult$means[2,])^2)
    if (dis11+dis22 < dis12+dis21){
      test_label[which(mgemResult$label==1)]=-1
    } else {
      test_label[which(mgemResult$label==2)]=-1
    }
    acc=length(which(test_label==testData$label))/dim(testData)[1]
    acc
  }
}

## performance test with different models
## 80% of the image 1 data as the training set
## 20% of the left data from image 1 as test data set 1
## image 2 as test data set 2
## image 3 as test data set 3
## data with s are standardized data

methods=c('glm', 'lda', 'qda', 'rf', 'qda_kmeans', 'lda_kmeans', 'qda_mg')
acc_comp=matrix(0,6,7)
colnames(acc_comp) = c('glm', 'lda', 'qda', 'rf', 'qda_kmeans', 'lda_kmeans', 'qda_mg')
rownames(acc_comp) = c('Original_1', 'Original_2','Original_2', 'Standardized_1', 'Standardized_2', 'Standardized_3')
for (i in 1:6){
  for (j in 1:7){
    if (j >4 & i <4){     
    }else{
      acc_comp[i,j]=AccRate(trainData[[ceiling(i/3)]], testData[[i]], methods[j])
    }
  }
}

save(acc_comp, file="acc_comp.rdata")

## cross validations
image.directory <- file.path("/accounts/class/s215a/s215a-9/lab4")
#image.directory <- file.path("/Users/rzdream/Dropbox/2014Fall/STAT215/lab4/image_data")
#import image files and append data row-wise
colnames <- c('y','x','label','NDAI','SD','CORR','DF','CF','BF','AF','AN')
import.data <- function(path,col.names) {
  setwd(path)
  image.files <- list.files(pattern = "[.]txt$")
  data <- foreach(i = image.files, .combine = rbind) %dopar%{
    cbind(read.table(i,header=FALSE,col.names=colnames),image=i)
  }
  data$image <- sub('.txt','',data$image,fixed=TRUE)
  return(data)
}
idata <- import.data(image.directory,colnames)
idata2 <- idata[idata$label!=0,]
idataS <- idata2
idataS[,4:11] <- idataS[,4:11]/(matrix(1,dim(idata2)[1],1)%*%colSds(as.matrix(idata2[,4:11])))
K=10
repeated=3
ndata=length(idata2[,1])
acc_comp2=matrix(0,2,7)
for (r in 1:repeated){
  idx=createFolds(1:ndata, K)
  aa = foreach (k =1:K, .combine=rbind) %dopar%{
    tempacc=matrix(0,2,7)
    trainData=idata2[idx[[k]],]
    testData=idata2[-idx[[k]],]
    trainDataS=idataS[idx[[k]],]
    testDataS=idataS[-idx[[k]],]
    for (j in 1:7){
      if (j<7)
        tempacc[1,j]=AccRate(trainData, testData, methods[j])
      tempacc[2,j]=AccRate(trainDataS, testDataS, methods[j])
    }
    return (tempacc)
  }
  for (k in 1:K){
    acc_comp2=acc_comp2+aa[(k*2-1):(k*2),]
  }
}
acc_comp2=acc_comp2/repeated
save(acc_comp2, file="acc_comp2.rdata")
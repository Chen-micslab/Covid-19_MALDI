data = read.csv("./data/97_feature_transform.csv",header = F,row.names = 1)
data1 = data
a = data[1,] 
a1 = character()
for (i in 1:length(a)){
  a1 = c(a1,a[1,i])
}
b1 = rownames(data)
num1 = 0
for(i in 1:length(a1)){
  if (a[i]==a[1]){
    num1 = num1 + 1
  }
}

mean1 = numeric()
mean2 = numeric()
for (i in 2:length(b1)){ 
  fc = mean(as.numeric(data[i,1:num1]))/mean(as.numeric(data[i,(num1+1):length(a1)]))
  mean1 = c(mean1,mean(as.numeric(data[i,1:num1])))   
  mean2 = c(mean2,mean(as.numeric(data[i,(num1+1):length(a1)])))
}


FC = numeric()
log2FC = numeric()
for (i in 2:length(b1)){
  fc = mean(as.numeric(data[i,1:num1]))/mean(as.numeric(data[i,(num1+1):length(a1)]))
  FC = c(FC,fc)
  log2FC = c(log2FC,log2(fc))
}

P_w = numeric()
for (i in 2:length(b1)){
  y = wilcox.test(as.numeric(data[i,1:num1]),as.numeric(data[i,(num1+1):length(data)]),alternative ='two.sided',paired = FALSE )
  pvalue = y$p.value
  P_w = c(P_w,pvalue)
}
pad_w = p.adjust(P_w,method = 'BH')
data2 = data.frame('mean1'=mean1,'mean2'=mean2,'FC'=FC,'log2FC'=log2FC,'独立wilcoxon p-value'=P_w,'独立wilcoxon adjust-p'=pad_w)
write.csv(data2,"./data/97_feature_P&FC.CSV",row.names = F)

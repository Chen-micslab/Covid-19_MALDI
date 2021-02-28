data = read.csv('./data/219_feature_no_fill_gap.csv',header = 1,row.names = 1)
a = row.names(data)
data1 = data 
for (i in 1:length(a)){
  b = as.numeric(data[i,])
  d = as.numeric()
  for (j in 1:length(b)){
    if (b[j]!=0){
      d = c(d,b[j])
    }
  }
  gap = min(d)/10
  for (k in 1:length(b)){
    if (b[k]==0){
      data1[i,k]=gap
    }
  }
}
write.csv(data1,"./data/219_feature_fill_gap.csv")


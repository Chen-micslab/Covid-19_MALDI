data = read.csv("E:/r work/2019 cov wan/data/R语言 峰对齐过滤后数据/0-1000 new/0-1000 new detected peaks fill gap top97 转置.CSV",
                header = F,row.names = 1)
data1 = data
#提取数据第一行存入a1，即为分类情况
a = data[1,] 
a1 = character()
for (i in 1:length(a)){
  a1 = c(a1,a[1,i])
}
#提取行名存入b1，即为feature的种类
b1 = rownames(data)

#求case样本的数量
num1 = 0
for(i in 1:length(a1)){
  if (a[i]==a[1]){
    num1 = num1 + 1
  }
}

P_w = numeric()
#wilcoxon-test
for (i in 2:length(b1)){
  y = wilcox.test(as.numeric(data[i,1:num1]),as.numeric(data[i,(num1+1):length(data)]),
                  alternative ='two.sided',paired = FALSE )
  pvalue = y$p.value
  P_w = c(P_w,pvalue)
}
p_adjust = p.adjust(P_w,method = 'BH')
x = cbind(b1[2:length(b1)],P_w,p_adjust)
write.csv(x,'E:/r work/2019 cov wan/data/R语言 峰对齐过滤后数据/0-1000 new/0-1000 new detected peaks fill gap adjust p.CSV')


#正态性分布检验
data = read.csv("E:/r work/2019 cov wan/data/R语言 峰对齐过滤后数据/0-1000 new/choose feature/Top 10 转置.CSV",
                header = F,row.names = 1)

ZT = numeric()
num1 = 0
for(i in 1:length(a1)){
  if (a[i]==a[1]){
    num1 = num1 + 1
  }
}
for(i in 2:length(data[,1])){
  y = as.numeric(data[i,(num1+1):length(data)])
  q = shapiro.test(y)
  ZT = c(ZT,q$p.value)
}
data3 = data.frame(ZT)







#####用3s方法去除异常值
data = read.csv("E:/r work/2019 cov wan/data/R语言 峰对齐过滤后数据/0-1000 new/choose feature/Top 10 转置.CSV",
                header = F,row.names = 1)

#calculate the number of case samples
num1 = 0
a = data[1,] 
a1 = character()
for (i in 1:length(a)){
  a1 = c(a1,a[1,i])
} 
for(i in 1:length(a1)){
  if (a[i]==a[1]){
    num1 = num1 + 1
  }
}
#remove the abnormal value in control samples, and calculate the adjust p and FC of 97 features
p_w = as.numeric()
FC = as.numeric()
for (j in 2:length(data[,1])){
print(paste('num:',j))
y = as.numeric(data[j,(num1+1):length(data)])
sd1 = 3*sd(y)
mean1 = mean(y)
y_new_control = as.numeric()
for (i in 1:length(y)){
  y1 = abs(y[i]-mean1)
  if(y1 <sd1){
    y_new_control = c(y_new_control,y[i])
  }
}

y = as.numeric(data[j,1:num1])
sd1 = 3*sd(y)
mean1 = mean(y)
y_new_case = as.numeric()
for (i in 1:length(y)){
  y1 = abs(y[i]-mean1)
  if(y1<sd1){
    y_new_case = c(y_new_case,y[i])
  }
}
#calculate log2 FC
fc = mean(y_new_case)/mean(y_new_control)
FC = c(FC,log2(fc))
#calculate p value of feature using wilcoxon test
y_w = wilcox.test(y_new_case,y_new_control,alternative ='two.sided',paired = FALSE )
pvalue = y_w$p.value
p_w = c(p_w,pvalue)
print(length(y_new_case))
print(length(y_new_control))
}
p_adjust = p.adjust(p_w,method = 'BH')
data_out = data.frame('FC'=FC,'P-value'=p_w,'P-adjust'=p_adjust)
write.csv(data_out,'E:/r work/2019 cov wan/data/R语言 峰对齐过滤后数据/0-1000 new/0-1000 new detected peaks fill gap adjust p extract abnormal.CSV')



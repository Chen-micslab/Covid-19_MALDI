from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

data1_1 = pd.read_csv('./data/97_feature.csv',header=0)
data1 = np.array(data1_1)
np.random.seed(99)
permutation = np.random.permutation(data1[:,0].shape[0])
data1 = data1[permutation]
x = data1[:,1:]
y = data1[:,0]
from sklearn.preprocessing import StandardScaler
st = StandardScaler().fit(x)
x1 = st.transform(x)
pca = PCA(n_components=0.99)
pca.fit(x1)
x1 = pca.transform(x1)
st = StandardScaler().fit(x)
x2 = st.transform(x)
sen = []
spe = []
acc = []
c_m = [179, 19, 17, 21, 7]
g_m = [7, 95, 97, 99, 98]
c = [2, 2, 2, 2, 3]
g = [0.3, 0.15, 0.15, 0.2, 0.2]
c_1 = [300, 300, 300, 300, 300]
c_2 = [45, 45, 45, 45, 45]
c_3 = [40, 40, 40, 40, 40]
skf = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x1, y)
for k,(train, test) in enumerate(skf):
    skf1 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
    for j, (a, b) in enumerate(skf1):
        if j == k:
            x_train = x[a]
            x_test = x[b]
    skf2 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x2, y)
    for j, (a, b) in enumerate(skf2):
        if j == k:
            x_train1 = x2[a]
            x_test1 = x2[b]
    svm = SVC(C=c_m[k], gamma=g_m[k]/100000,probability=True).fit(x1[train], y[train])
    y_pre1 = svm.predict(x1[train])
    RF = XGBClassifier(n_estimators=500,max_depth=c[k],learning_rate=g[k]).fit(x_train,y[train])
    y_pre2 = RF.predict(x_train)
    mlp = MLPClassifier(hidden_layer_sizes=(c_1[k], c_2[k], c_3[k]), solver='adam', random_state=1, max_iter=10000).fit(x_train1, y[train])
    y_pre3 = mlp.predict(x_train1)
    y_pre = np.column_stack([y_pre1,y_pre2,y_pre3])
    y_pre1_1 = svm.predict(x1[test])
    y_pre2_1 = RF.predict(x_test)
    y_pre3_1 = mlp.predict(x_test1)
    y_pre_1 = np.column_stack([y_pre1_1, y_pre2_1,y_pre3_1])
    meta_learn = RandomForestClassifier(n_estimators=600, max_depth=8, random_state=11).fit(y_pre, y[train])
    y_p = meta_learn.predict(y_pre_1)
    cn = confusion_matrix(y[test], y_p)
    print(cn)
    sen1 = cn[0,0]/(cn[0,0]+cn[0,1])
    spe1 = cn[1,1]/(cn[1,1]+cn[1,0])
    acc1 = (cn[0,0]+cn[1,1])/(cn[0,0]+cn[0,1]+cn[1,1]+cn[1,0])
    spe.append(spe1)
    sen.append(sen1)
    acc.append(acc1)
print('accuracy:',np.mean(acc),'std:',np.std(acc))
print('sensitivity:',np.mean(sen),'std:',np.std(sen))
print('specificity:',np.mean(spe),'std:',np.std(spe))

aucs = []
hold = np.linspace(0, 1, 100)
mean_fpr = np.linspace(0, 1, 1000)
tprs = []
skf = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
for k,(train, test) in enumerate(skf):
    skf1 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
    for j, (a, b) in enumerate(skf1):
        if j == k:
            x_train = x[a]
            x_test = x[b]
    skf2 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x2, y)
    for j, (a, b) in enumerate(skf2):
        if j == k:
            x_train1 = x2[a]
            x_test1 = x2[b]
    svm = SVC(C=c_m[k], gamma=g_m[k] / 100000, probability=True).fit(x1[train], y[train])
    y_pre1 = svm.predict(x1[train])
    RF = XGBClassifier(n_estimators=500, max_depth=c[k], learning_rate=g[k]).fit(x_train, y[train])
    y_pre2 = RF.predict(x_train)
    mlp = MLPClassifier(hidden_layer_sizes=(c_1[k], c_2[k], c_3[k]), solver='adam', random_state=1, max_iter=10000).fit(
        x_train1, y[train])
    y_pre3 = mlp.predict(x_train1)
    y_pre = np.column_stack([y_pre1, y_pre2,y_pre3])  # 合并三个模型的训练集的预测结果作为新模型的训练集
    y_pre1_1 = svm.predict(x1[test])
    y_pre2_1 = RF.predict(x_test)
    y_pre3_1 = mlp.predict(x_test1)
    y_pre_1 = np.column_stack([y_pre1_1, y_pre2_1,y_pre3_1])  # 合并三个模型的测试集预测结果作为新模型的测试集
    meta_learn = RandomForestClassifier(n_estimators=600, max_depth=8, random_state=11).fit(y_pre, y[train])
    y_pro = meta_learn.predict_proba(y_pre_1)
    y_last = np.zeros((len(y[test]),))
    tpr = []
    fpr = []
    for h in hold:
        for j in range(len(y[test])):
          if y_pro[j,1] > h :
              y_last[j] = 1
          else:
              y_last[j] = 0
        cn = confusion_matrix(y[test], y_last)
        tpr1 = cn[0,0] / (cn[0,0] + cn[0,1])
        fpr1 = cn[1,0] / (cn[1,0] + cn[1,1])
        tpr.append(tpr1)
        fpr.append(fpr1)
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
tprs = np.array(tprs)
fig, ax = plt.subplots()
ax.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='-', lw=1.5, color='gray',alpha=0.5)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
print('mean roc auc:',mean_auc,'std:',std_auc)
font1 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 28,}
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 24,}
ax.plot(mean_fpr, mean_tpr, color='#990000',
        label='AUC = %0.2f' % (mean_auc),
        lw=3, alpha=0.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
plt.yticks(fontproperties = 'Arial', size = 26)
plt.xticks(fontproperties = 'Arial', size = 26)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.tick_params(width=2)
ax.legend(loc="lower right",prop = font2,frameon = False)
plt.xlabel('1 - Specificity',font1,labelpad=6)
plt.ylabel('Sensitivity',font1,labelpad=10)

aucs = []
hold = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 1000)
precs = []

skf = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
for k,(train, test) in enumerate(skf):
    skf1 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
    for j, (a, b) in enumerate(skf1):
        if j == k:
            x_train = x[a]
            x_test = x[b]
    skf2 = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x2, y)
    for j, (a, b) in enumerate(skf2):
        if j == k:
            x_train1 = x2[a]
            x_test1 = x2[b]
    svm = SVC(C=c_m[k], gamma=g_m[k] / 100000, probability=True).fit(x1[train], y[train])
    y_pre1 = svm.predict(x1[train])
    RF = XGBClassifier(n_estimators=500, max_depth=c[k], learning_rate=g[k]).fit(x_train, y[train])
    y_pre2 = RF.predict(x_train)
    mlp = MLPClassifier(hidden_layer_sizes=(c_1[k], c_2[k], c_3[k]), solver='adam', random_state=1, max_iter=10000).fit(
        x_train1, y[train])
    y_pre3 = mlp.predict(x_train1)
    y_pre = np.column_stack([y_pre1, y_pre2,y_pre3])
    y_pre1_1 = svm.predict(x1[test])
    y_pre2_1 = RF.predict(x_test)
    y_pre3_1 = mlp.predict(x_test1)
    y_pre_1 = np.column_stack([y_pre1_1, y_pre2_1,y_pre3_1])
    meta_learn = RandomForestClassifier(n_estimators=600, max_depth=8, random_state=11).fit(y_pre, y[train])
    y_pro = meta_learn.predict_proba(y_pre_1)
    y_last = np.zeros((len(y[test]),))
    recall = []
    prec = []
    for h in hold:
        for j in range(len(y[test])):
          if y_pro[j,1] > h :
              y_last[j] = 1
          else:
              y_last[j] = 0
        cn = confusion_matrix(y[test], y_last)
        recall1 = cn[0,0] / (cn[0,0] + cn[0,1])
        if (cn[0,0]==0)and(cn[1,0]==0):
            prec1 = 1
        else:
            prec1 = cn[0,0] / (cn[0,0] + cn[1,0])
        recall.append(recall1)
        prec.append(prec1)
    pr_auc = auc(recall,prec)
    aucs.append(pr_auc)
    interp_prec = np.interp(mean_recall, recall, prec)
    interp_prec[0] = 1
    precs.append(interp_prec)
precs= np.array(precs)
fig1, ax1 = plt.subplots()
ax1.plot([-0.05, 1.05], [0.33, 0.33], linestyle='-', lw=1.5, color='gray',
         alpha=0.5)
mean_prec = np.mean(precs, axis=0)
mean_prec[-1] = 0.333
mean_auc = auc(mean_recall, mean_prec)
std_auc = np.std(aucs)
print('mean PR auc:',mean_auc,'std:',std_auc)
ax1.plot(mean_recall, mean_prec, color='#990000',lw=3, alpha=0.8,
         label='AUC = %0.2f' % (mean_auc))
font1 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 28,}
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 24,}
ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
plt.yticks(fontproperties = 'Arial', size = 26)
plt.xticks(fontproperties = 'Arial', size = 26)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['right'].set_linewidth(2)
ax1.tick_params(width=2)
ax1.legend(loc="lower right",prop = font2,frameon = False)
plt.xlabel('Recall',font1,labelpad=6)
plt.ylabel('Precision',font1,labelpad=10)
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 300
plt.show()
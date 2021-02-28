from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import  sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

data1_1 = pd.read_csv('./data/219_feature.csv',header=0)
data1 = np.array(data1_1)
np.random.seed(99)
permutation = np.random.permutation(data1[:,0].shape[0])
data1 = data1[permutation]
x = data1[:,1:]
y = data1[:,0]
from sklearn.preprocessing import StandardScaler
st = StandardScaler().fit(x)
x = st.transform(x)
pca = PCA(n_components=0.99)
pca.fit(x)
x = pca.transform(x)
c_m = []
g_m = []
skf=StratifiedKFold(n_splits=5,random_state=30,shuffle=True).split(x,y)
for k,(train, test) in enumerate(skf):
    print(k)
    g = 0
    h = [0,0]
    for j in range(1,20):
        for l in ['uniform','distance']:
            b = []
            for state in range(10):
                acd =KNeighborsClassifier(n_neighbors=j,weights=l)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                scores = cross_val_score(acd, x[train], y[train], cv=kfold, scoring='roc_auc', n_jobs=6)
                b.append(scores.mean())
            c = np.mean(b)
            if c > g:
                g = c
                h[0] = j
                h[1] = l
    c_m.append(h[0])
    g_m.append(h[1])
print(c_m)
print(g_m)

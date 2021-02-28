import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import time

data1_1 = pd.read_csv('./data/97_feature.csv',header=0)
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
skf=StratifiedKFold(n_splits=5,random_state=30,shuffle=True).split(x,y)
c_m = []
g_m = []
for k,(train, test) in enumerate(skf):
    g = 0
    h = [0, 0]
    for j in range(1, 200, 2):
        for l in range(1, 100):
            b = []
            for state in range(10):
                acd = SVC(C=j, gamma=l / 100000)
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train],y[train])
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

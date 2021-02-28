import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import  PCA
from sklearn.model_selection import StratifiedKFold

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
diff = []
a = []
skf = StratifiedKFold(n_splits=5, random_state=30, shuffle=True).split(x, y)
c_1 = []
c_2 = []
c_3 = []
for k,(train, test) in enumerate(skf):
    print(k)
    g = 0
    h = [0, 0, 0]
    for i in [ 79,80,81 ]:
        for j in [ 79,80,81 ]:
            for m in [50]:
                b = []
                for state in range(10):
                    acd = MLPClassifier(hidden_layer_sizes=(i,j,m), solver='adam', random_state=1, max_iter=10000)
                    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=state).split(x[train], y[train])
                    scores = cross_val_score(acd, x[train], y[train], cv=kfold, scoring='roc_auc', n_jobs=3)
                    b.append(scores.mean())
                c = np.mean(b)
                if c > g:
                    g = c
                    h[0] = j
                    h[1] = j
                    h[2] = m
    c_1.append(h[0])
    c_2.append(h[1])
    c_3.append(h[2])
print(c_1)
print(c_2)
print(c_3)

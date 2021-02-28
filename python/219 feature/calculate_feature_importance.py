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

data1_1 = pd.read_csv('./data/219_feature.csv',header=0)
data1 = np.array(data1_1)
im1 = np.zeros((97,))
for i in range(10):
    data1 = np.array(data1_1)
    print(i)
    np.random.seed(i)
    permutation = np.random.permutation(data1[:,0].shape[0])
    data1 = data1[permutation]
    x = data1[:,1:]
    y = data1[:,0]
    sen = []
    spe = []
    acc = []
    a = []
    RF = RandomForestClassifier(n_estimators=500,random_state=1).fit(x, y)
    im = RF.feature_importances_
    print(im)
    im1 = im1+im
im1 = im1/10
np.savetxt('./data/RF feature importance.csv',im1,delimiter=',')

im1 = np.zeros((97,))
for i in range(10):
    data1 = np.array(data1_1)
    print(i)
    np.random.seed(i)
    permutation = np.random.permutation(data1[:,0].shape[0])
    data1 = data1[permutation]
    x = data1[:,1:]
    y = data1[:,0]
    sen = []
    spe = []
    acc = []
    a = []
    xgb = XGBClassifier(n_estimators=500,random_state=1).fit(x, y)
    im = xgb.feature_importances_
    im1 = im1+im
im1 = im1/10
np.savetxt('./data/XGB feature importance.csv',im1,delimiter=',')
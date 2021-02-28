import umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numba
import random
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import MultipleLocator

data = pd.read_csv('./data/219_feature.csv',header=0)
data_fea = data.iloc[:, 1:]
st = StandardScaler().fit(data_fea)
data_fea = st.transform(data_fea)
color = []
for i in range(len(data)):
        color.append(data.iloc[i,0])
classes = ['Asymptomatic','Healthy']
reducer = umap.UMAP(n_neighbors=14, min_dist=0.5, n_components=5,random_state=44)
embedding = reducer.fit_transform(data_fea)
print(embedding.shape)
fig1, ax1 = plt.subplots()
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 24,}
# plt.scatter(embedding[:, 0], embedding[:, 1], c=color,cmap='Spectral', s=40)
a = plt.scatter(embedding[:, 0], embedding[:, 1], c=color,cmap='Spectral', s=40)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24)
x_major_locator=MultipleLocator(0.9)
y_major_locator=MultipleLocator(0.8)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_major_locator(y_major_locator)
print(a.legend_elements()[0])
plt.legend(handles = a.legend_elements()[0], labels=classes,prop = font2,bbox_to_anchor=(0.9, 0.8), loc=3, borderaxespad=0)
plt.gca().set_aspect('equal', 'datalim')
plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 200
plt.show()
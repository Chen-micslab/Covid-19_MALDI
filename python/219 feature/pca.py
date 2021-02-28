import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.pyplot import MultipleLocator

data = pd.read_csv('./data/219_feature.csv',header=0)
X = data.iloc[:, 1:]
y = data.iloc[:,0]
print(X.shape)
color = []
for i in range(len(data)):
        color.append(data.iloc[i,0])
pca = PCA(n_components=0.9)
X_p = pca.fit(X).transform(X)
ratio = pca.explained_variance_ratio_
print(ratio[1])
print(X_p.shape)
fig1, ax1 = plt.subplots()
classes = ['Asymptomatic','Healthy']
font1 =  {'family' : 'Arial',
'weight' : 'normal',
'size' : 22,
}
font2 = {'family' : 'Arial',
'weight' : 'book',
'size'   : 24,}
a = plt.scatter(X_p[:, 0], X_p[:, 1], c=color,cmap='Spectral', s=40)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.yticks(fontproperties = 'Arial', size = 24)
plt.xticks(fontproperties = 'Arial', size = 24)
x_major_locator=MultipleLocator(200)
y_major_locator=MultipleLocator(130)
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_major_locator(y_major_locator)
ax1.set(ylim=[-300, 400])
print(a.legend_elements()[0])
plt.legend(handles = a.legend_elements()[0], labels=classes,prop = font2,bbox_to_anchor=(0.9, 0.8), loc=3, borderaxespad=0)
plt.gca().set_aspect('equal', 'datalim')
plt.rcParams['figure.figsize'] = (8,8)
plt.xlabel('Dim1({0:0.1f}%)'.format(100*ratio[0]),font2)
plt.ylabel('Dim2({0:0.1f}%)'.format(100*ratio[1]),font2)
plt.rcParams['savefig.dpi'] = 2000
plt.rcParams['figure.dpi'] = 300
plt.show()
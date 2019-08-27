



#%%
from sklearn.datasets import make_moons
dataset=make_moons(n_samples=5000)
X=dataset[0]
y=dataset[1]

#%%
import numpy as np
X_centered=X-X.mean(axis=0)
U,s,V=np.linalg.svd(X_centered)
c1=V.T[:,0]
c2=V.T[:,1]
print(c1,c2)
W2=V.T[:,:2]
X2D=X_centered.dot(W2)
#%%
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X2D=pca.fit_transform(X)
print(pca.components_.T[:,0])
print(pca.explained_variance_ratio_)
#%%
pca=PCA()
pca.fit(X)
cumsum=np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>=0.95)+1
print(d)
pca=PCA(n_components=0.95)
X_reduced=pca.fit_transform(X)
#%%
pca=PCA(n_components=154)
X_mnist_reduced=pca.fit_transform(X_mnist)
X_mnist_recovered=pca.inverse_transform(X_mnist_reduced)
#%%
from sklearn.decomposition import IncrementalPCA
n_batches=100
inc_pca=IncrementalPCA(n_components=2)
for X_batch in np.array_split(X_mnist,n_batches):
inc_pca.partial_fit(X_batch)
X_mnist_reduced=inc_pca.transform(X_mnist)
#%%
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
import numpy as np
clf=Pipeline([
    ('kernel',KernelPCA(n_components=2)),
    ('line',LinearRegression())
])
param_grid=[{
    'kernel_gamma':np.linspace(0.03,0.05,10),
    'kpca_kernel':['rbf','sigmoid']
}]
grid_search=GridSearchCV(clf,param_grid,cv=3)
grid_search.fit(X,y)
#%%
print(grid_search.best_params_)
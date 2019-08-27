#%%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

iris=load_iris()
iris.keys()

#%%
X=iris['data'][:,(2,3)]
y=iris['target']
X.shape
y=(y==2).astype(np.float64)
svm_clf=Pipeline((
    ('scaler',StandardScaler()),
    ('svm',LinearSVC(C=1,loss='hinge'))
))
svm_clf.fit(X,y)
svm_clf.predict([[5.5,1.7]])
# 不同于 Logistic 回归分类器，SVM 分类器不会输出每个类别的概率。
#%%
# 作为一种选择，你可以在 SVC 类，使用 SVC(kernel="linear", C=1)  ，但是它比较慢，尤其在
# 较大的训练集上，所以一般不被推荐。另一个选择是使用 SGDClassifier  类，
# 即 SGDClassifier(loss="hinge", alpha=1/(m*C))  。它应用了随机梯度下降（SGD 见第四章）
# 来训练一个线性 SVM 分类器。尽管它不会和 LinearSVC  一样快速收敛，但是对于处理那些不
# 适合放在内存的大数据集是非常有用的，或者处理在线分类任务同样有用。

#%%
# 非线性支持向量机分类
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

dataset=make_moons(n_samples=500)
X=dataset[0]
y=dataset[1]
svm=Pipeline((
    ('poly',PolynomialFeatures(degree=3)),
    ('scaler',StandardScaler()),
    ('svm',LinearSVC(C=10,loss='hinge'))
))
svm.fit(X,y)
pred0=svm.predict(X[y==0])
pred1=svm.predict(X[y==1])
#%%
from matplotlib import pyplot as plt
plt.scatter(x=X[y==0,0],y=X[y==0,1],c='yellow')
plt.scatter(x=X[y==1,0],y=X[y==1,1],c='red')
plt.show()

#%%
# 加载包
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# 画出数据点和边界
def border_of_classifier(sklearn_cl, x, y):
        """
        param sklearn_cl : skearn 的分类器
        param x: np.array 
        param y: np.array
        """
        ## 1 生成网格数据
        x_min, y_min = x.min(axis = 0) - 1
        x_max, y_max = x.max(axis = 0) + 1
        # 利用一组网格数据求出方程的值，然后把边界画出来。
        x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01))
        # 计算出分类器对所有数据点的分类结果 生成网格采样
        mesh_output = sklearn_cl.predict(np.c_[x_values.ravel(), y_values.ravel()])
        # 数组维度变形  
        mesh_output = mesh_output.reshape(x_values.shape)
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ## 会根据 mesh_output结果自动从 cmap 中选择颜色
        plt.pcolormesh(x_values, y_values, mesh_output, cmap = 'rainbow')
        plt.scatter(x[:, 0], x[:, 1], c = y, s=100, edgecolors ='steelblue' , linewidth = 1, cmap = plt.cm.Spectral)
        plt.xlim(x_values.min(), x_values.max())
        plt.ylim(y_values.min(), y_values.max())
        # 设置x轴和y轴
        plt.xticks((np.arange(np.ceil(min(x[:, 0]) - 1), np.ceil(max(x[:, 0]) + 1), 1.0)))
        plt.yticks((np.arange(np.ceil(min(x[:, 1]) - 1), np.ceil(max(x[:, 1]) + 1), 1.0)))
        plt.show()

border_of_classifier(svm,X,y)

#%%
x_value, y_value = np.meshgrid(np.arange(0, 5, 1),np.arange(0, 3, 1))
print(x_value.shape)
print(y_value)
print(np.c_[x_value.ravel(), y_value.ravel()])
#%%
# 多项式核
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, y)

#%%
border_of_classifier(poly_kernel_svm_clf,X,y)

#%%
# 增加相似特征
# 另一种解决非线性问题的方法是使用相似函数（similarity funtion）计算每个样本与特定地标
# （landmark）的相似度。例如，让我们来看看前面讨论过的一维数据集，并
# 在 x1=-2  和 x1=1  之间增加两个地标（图 5-8 左图）。接下来，我们定义一个相似函数，即高
# 斯径向基函数（Gaussian Radial Basis Function，RBF），设置 γ = 0.3 
rbf_kernel_svm_clf = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)

#%%
# 训练数据需要被中心化和标准化
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

#%%
from sklearn.svm import SVR
svm_ploy_reg=SVR(degree=2,kernel='poly',C=100,epsilon=0.1)
svm_ploy_reg.fit(X,y)

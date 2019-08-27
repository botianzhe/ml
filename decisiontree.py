#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris=load_iris()
X=iris['data'][:,2:]
y=iris['target']

#%%
tree=DecisionTreeClassifier(max_depth=2)
tree.fit(X,y)

#%%
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='aaa.dot',feature_names=iris.feature_names[2:],class_names=iris.target_names,rounded=True,filled=True)

#%%
# 决策树的众多特性之一就是， 它不需要太多的数据预处理， 尤其是不需要进行特征的缩
# 放或者归一化。
import numpy as np
import matplotlib.pyplot as plt
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

border_of_classifier(tree,X,y)


#%%
tree.predict_proba([[5,1.5]])
tree.predict([[5,1.5]])
#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

#%%
export_graphviz(tree_reg,out_file='aaa2.dot',feature_names=iris.feature_names[2:],class_names=iris.target_names,rounded=True,filled=True)

#%%

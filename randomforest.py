#%%
# 令人惊奇的是这种投票分类器得出的结果经常会比集成中最好的一个分类器结果更好。事实
# 上，即使每一个分类器都是一个弱学习器（意味着它们也就比瞎猜好点），集成后仍然是一
# 个强学习器（高准确率），只要有足够数量的弱学习者，他们就足够多样化。

#%%
# 如果使每一个分类器都独立自主的分类，那么集成模型会工作的很好。去得到多样的分类器
# 的方法之一就是用完全不同的算法，这会使它们会做出不同种类的错误，这会提高集成的正
# 确率
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
dataset=make_moons(n_samples=5000,noise=0.5)
X=dataset[0]
y=dataset[1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
log_clf=LogisticRegression()
rnd_clf=RandomForestClassifier()
svm_clf=SVC(probability=True)
voting_clf=VotingClassifier(estimators=[('lr',log_clf),('rnd',rnd_clf),('svm',svm_clf)],voting='hard')
voting_clf.fit(X_train,y_train)

#%%
from sklearn.metrics import accuracy_score
y_pred=voting_clf.predict(X_test)
accuracy_score(y_test,y_pred)

#%%
for clf in [log_clf,rnd_clf,svm_clf]:
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))

#%%
# 接下来的代码训练了一个 500 个决策树分类器的集成，每一个都
# 是在数据集上有放回采样 100 个训练实例下进行训练（这是 Bagging 的例子，如果你想尝试
# Pasting，就设置 bootstrap=False  ）。 n_jobs  参数告诉 sklearn 用于训练和预测所需要 CPU
# 核的数量。（-1 代表着 sklearn 会使用所有空闲核）：
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=42)
tree.fit(X_train,y_train)
print(tree.score(X_test,y_test))
bag_clf=BaggingClassifier(DecisionTreeClassifier(random_state=42),n_estimators=500,max_samples=100,bootstrap=True,n_jobs=-1,oob_score=True)
bag_clf.fit(X_train,y_train)
print(bag_clf.oob_score_)
print(bag_clf.score(X_test,y_test))

#%%
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

border_of_classifier(tree,X_train,y_train)

#%%
# 在 sklearn 中，你可以在训练后需要创建一个 BaggingClassifier  来自动评估时设
# 置 oob_score=True  来自动评估。接下来的代码展示了这个操作。评估结果通过变
# 量 oob_score_  来显示：
bag_clf=BaggingClassifier(DecisionTreeClassifier(random_state=42),n_estimators=500,max_samples=100,bootstrap=True,n_jobs=-1,oob_score=True)
bag_clf.fit(X_train,y_train)
print(bag_clf.oob_score_)


#%%
# 让我们通过一个使用决策树当做基分类器的简单的回归例子（回归当然也可以使用梯度提
# 升）。这被叫做梯度提升回归树（GBRT，Gradient Tree Boosting 或者 Gradient Boosted
# Regression Trees）。首先我们用 DecisionTreeRegressor  去拟合训练集（例如一个有噪二次
# 训练集）：
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2=y-tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
y3=y2-tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)
y_pred=sum(clf.predict(X) for clf in (tree_reg1,tree_reg2,tree_reg3))
res=(y_pred>=0.5).astype('int')
print(accuracy_score(y,res))

#%%
# 我们可以使用 sklean 中的 GradientBoostingRegressor  来训练 GBRT 集成。
# 与 RandomForestClassifier  相似，它也有超参数去控制决策树的生长（例
# 如 max_depth  ， min_samples_leaf  等等），也有超参数去控制集成训练，例如基分类器的数
# 量（ n_estimators  ）。接下来的代码创建了与之前相同的集成：
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X,y)
y_pred=gbrt.predict(X)
res=(y_pred>=0.5).astype('int')
print(accuracy_score(y,res))
#%%
# 为了找到树的最优数量，你可以使用早停技术（第四章讨论）。最简单使用这个技术的方法
# 就是使用 staged_predict()  ：它在训练的每个阶段（用一棵树，两棵树等）返回一个迭代
# 器。加下来的代码用 120 个树训练了一个 GBRT 集成，然后在训练的每个阶段验证错误以找
# 到树的最佳数量，最后使用 GBRT 树的最优数量训练另一个集成：
from sklearn.metrics import mean_squared_error
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train,y_train)
errors=[mean_squared_error(y_test,pred) for pred in gbrt.staged_predict(X_test)]
bst_n_estimators=np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X,y)
y_pred=gbrt_best.predict(X)
res=(y_pred>=0.5).astype('int')
print(accuracy_score(y,res))
#%%
# 也可以早早的停止训练来实现早停（与先在一大堆树中训练，然后再回头去找最优数目相
# 反）。你可以通过设置 warm_start=True  来实现 ，这使得当 fit()  方法被调用时 sklearn 保留
# 现有树，并允许增量训练。接下来的代码在当一行中的五次迭代验证错误没有改善时会停止
# 训练：
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break # early stopping
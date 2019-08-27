#%%
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',)

#%%
mnist['DESCR']
#%%
X,y=mnist['data'],mnist['target']

#%%
X.shape

#%%
%matplotlib inline
import matplotlib.pyplot as plt
data=X[0]
reshapedata=data.reshape(28,28)
plt.imshow(reshapedata ,cmap = matplotlib.cm.binary, interpolation="nearest")
print(y[0])
#%%
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]


#%%
# 打乱训练集
import numpy as np
random=np.random.permutation(60000)
X_train,y_train=X_train[random],y_train[random]
print(y_train[:100])
#%%
# 训练一个二分类器
y_train_5=(y_train=='5')
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(random_state=42)
sgd.fit(X_train,y_train_5)
sgd.predict([data])

#%%
#%% [markdown]
# 交叉验证

#%%
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds=StratifiedKFold(n_splits=3,random_state=42)
for train_index,test_index in skfolds.split(X_train,y_train_5):
    clone_sgd=clone(sgd)
    X_train_folds=X_train[train_index]
    y_train_folds=y_train_5[train_index]
    X_test_folds=X_train[test_index]
    y_test_folds=y_train_5[test_index]
    clone_sgd.fit(X_train_folds,y_train_folds)
    y_pred=clone_sgd.predict(X_test_folds)
    n_correct=sum(y_test_folds==y_pred)
    print(n_correct/len(y_pred))

#%%

from sklearn.model_selection import cross_val_score
cross_val_score(sgd,X_train,y_train_5,cv=3,scoring='accuracy')

#%%
from sklearn.base import BaseEstimator
class MyTrainer(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
my=MyTrainer()
cross_val_score(my,X_train,y_train_5,cv=3,scoring='accuracy')


#%%
# 混淆矩阵
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
pred=cross_val_predict(sgd,X_train,y_train_5,cv=3)
confusion_matrix(y_train_5,pred)

#%%
# 准确率与召回率
from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(y_train_5,pred))
print(recall_score(y_train_5,pred))
print(f1_score(y_train_5,pred))
#%%
# 准确率/召回率之间的折衷
y_score=sgd.decision_function(X_train)
print(y_score)
threshold = 0
y_some_digit_pred = (y_score > threshold)
y_some_digit_pred
#%%

pred=cross_val_predict(sgd,X_train,y_train_5,cv=3,method='decision_function')
from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_train_5,pred)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#%%
plt.plot(recalls[:-1],precisions[:-1])
plt.show()

#%%
# 假设你决定达到 90% 的准确率。你查阅第一幅图（放大一些），在 70000 附近找到一个
# 阈值。为了作出预测（目前为止只在训练集上预测）
pred=(y_score>1000)
precision_score(y_train_5,pred)
# recall_score(y_train_5,pred)

#%%
# ROC曲线
from sklearn.metrics import roc_curve

fpr,tpr,thres=roc_curve(y_train_5,y_score)
plt.plot(fpr,tpr)


#%%
# 一个比较分类器之间优劣的方法是：测量ROC曲线下的面积（AUC）。一个完美的分类器的
# ROC AUC 等于 1，而一个纯随机分类器的 ROC AUC 等于 0.5
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_score)

#%%
# 优先使用 PR 曲线当正例很少，或者当你关注假正例多于假
# 反例的时候。其他情况使用 ROC 曲线

#%%
# Scikit-Learn 可以探测出你想使用一个二分类器去完成多分类的任务，它会自动地执行OvA
sgd10=SGDClassifier()
sgd10.fit(X_train,y_train)


#%%
sgd10.predict([data])
scores=sgd10.decision_function([data])
np.argmax(scores)
sgd10.classes_
#%%
# 强制 Scikit-Learn 使用 OvO 策略或者 OvA 策略，你可以使用 OneVsOneClassifier  类
# 或者 OneVsRestClassifier  类。
from sklearn.multiclass import OneVsOneClassifier
ovo=OneVsOneClassifier(SGDClassifier(random_state=42))
ovo.fit(X_train,y_train)
ovo.predict([data])

#%%
ovo.decision_function([data])

#%%
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(X_train,y_train)
forest.predict_proba([data])
#%%
 cross_val_score(sgd10, X_train, y_train, cv=3, scoring="accuracy")

#%%
# 正则化
from sklearn.preprocessing import StandardScaler
scales=StandardScaler()
x_train_scaled=scales.fit_transform(X_train)
# cross_val_score(sgd10, x_train_scaled, y_train, cv=3, scoring="accuracy")
#%%
from sklearn.model_selection import cross_val_predict
y_pred=cross_val_predict(sgd10, x_train_scaled, y_train, cv=3)
confusionmatrix=confusion_matrix(y_train,y_pred)

plt.matshow(confusionmatrix,cmap=plt.cm.gray)
plt.show()

#%%
# 关注仅包含误差数据的图像呈现
row_sums=confusionmatrix.sum(axis=1,keepdims=True)
norm_confusion=confusionmatrix/row_sums
np.fill_diagonal(norm_confusion,0)#用 0 来填充对角线
plt.matshow(norm_confusion,cmap=plt.cm.gray)
plt.show()

#%%
# 画出3、5图
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_pred == cl_b)]
# 在一个块中画多个图
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(8,8))
plt.subplot(221)
plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222)
plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223)
plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224)
plot_digits(X_bb[:25], images_per_row=5)
plt.show()

#%%
# SGDClassifier  ，这是一个线性模型。它所做的全部工作就是分配一个类权重给每一个像
# 素，然后当它看到一张新的图片，它就将加权的像素强度相加，每个类得到一个新的值。所
# 以，因为 3 和 5 只有一小部分的像素有差异，这个模型很容易混淆它们。

# 减轻 3/5 混淆的一个方法是对图片进行预处理，确保它们都很好
# 地中心化和不过度旋转。这同样很可能帮助减轻其他类型的错误。

#%%
# 思考一个人脸识别器。如果对于同一张图片，它识别出几
# 个人，它应该做什么？当然它应该给每一个它识别出的人贴上一个标签。比方说，这个分类
# 器被训练成识别三个人脸，Alice，Bob，Charlie；然后当它被输入一张含有 Alice 和 Bob 的
# 图片，它应该输出 [1, 0, 1]  （意思是：Alice 是，Bob 不是，Charlie 是）。这种输出多个二
# 值标签的分类系统被叫做多标签分类系统。
from sklearn.neighbors import KNeighborsClassifier

y_large=(y_train.astype(np.int64)>=7)
y_ji=(y_train.astype(np.int64)%2==1)
print(y_large)
y=np.c_[y_large,y_ji]
model=KNeighborsClassifier()
model.fit(x_train_scaled,y)
model.predict([data])

#%%
# 有许多方法去评估一个多标签分类器，和选择正确的量度标准，这取决于你的项目。举个例
# 子，一个方法是对每个个体标签去量度 F1 值（或者前面讨论过的其他任意的二分类器的量度
# 标准），然后计算平均值。下面的代码计算全部标签的平均 F1 值：
y_pred=cross_val_predict(model,X_train,y,cv=3)
f1_score(y,y_pred,average='macro')

#%%
# 多输出分类
# 我们建立一个系统，它可以去除图片当中的噪音。它将一张混有噪音的图片
# 作为输入，期待它输出一张干净的数字图片，用一个像素强度的数组表示，就像 MNIST 图片
# 那样。注意到这个分类器的输出是多标签的（一个像素一个标签）和每个标签可以有多个值
# （像素强度取值范围从 0 到 255）。所以它是一个多输出分类系统的例子。
noise1=np.random.randint(0,100,(len(X_train),784))
noise2=np.random.randint(0,100,(len(X_test),784))
X_train_mod=X_train+noise1
X_test_mod=X_test+noise2
y_train_mod=X_train
y_test_mod=X_test
testdata=X_train_mod[0]
plt.imshow(testdata.reshape(28,28))

#%%
knn=KNeighborsClassifier()
knn.fit(X_train_mod,y_train_mod)

# 它是多标签分类的简单泛化，在这里每一个标签可以是多类别的（比如说，它可以有多于两个
# 可能值）。
# 为了说明这点，我们建立一个系统，它可以去除图片当中的噪音。它将一张混有噪音的图片
# 作为输入，期待它输出一张干净的数字图片，用一个像素强度的数组表示，就像 MNIST 图片
# 那样。注意到这个分类器的输出是多标签的（一个像素一个标签）和每个标签可以有多个值
# （像素强度取值范围从 0 到 255）。所以它是一个多输出分类系统的例子。

#%%
print(X_train_mod.shape)
print(testdata.shape)
clean=knn.predict([testdata])
print(clean)
plt.imshow(clean.reshape(28,28))


#%%
plt.imshow(X_train[0].reshape(28,28))

#%%

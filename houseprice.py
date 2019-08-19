import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
fetch_housing_data()

#%%
import pandas as pd
import os
HOUSING_PATH = "datasets/housing"
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
house=load_housing_data()
house.describe()

#%%
%matplotlib inline
import matplotlib.pyplot as plt
house.hist(bins=50,figsize=(30,20))
plt.show()

#%%
# 随机抽取测试集
import numpy as np
def split_train_test(data,radio):
        random=np.random.permutation(data)
        testsize=int(len(data)*radio)
        test=data[:testsize]
        train=data[testsize:]
        return train,test
train,test=split_train_test(house,0.2)
train.describe()
#%%
# 根据hash的最后一个字节大小抽取
import hashlib

def test_check(id,radio):
        return hashlib.md5(np.int64(id)).digest()[-1]<256*radio
def train_test_split(data,id,radio):
        ids=data[id]
        test=ids.apply(lambda id_:test_check(id_,radio))
        return data.loc[~test],data.loc[test]
house_with_id=house.reset_index()
train,test=train_test_split(house_with_id,"index",0.2)
len(test)
#%%
house_with_id['id']=house_with_id['longitude']*1000+house_with_id['latitude']
train,test=train_test_split(house_with_id,"id",0.2)

#%%
# 分层抽取
house_with_id['income_cat']=np.ceil(house_with_id['median_income']/1.5)
house_with_id['income_cat'].where(house_with_id['income_cat']<5,5.0,inplace=True)
house_with_id.head()
house_with_id['income_cat'].hist()
#%%
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train,test in split.split(house_with_id,house_with_id['income_cat']):
        train_set=house_with_id.loc[train]
        test_set=house_with_id.loc[test]
train_set['income_cat'].value_counts()/len(train_set)
for set in (train_set,test_set):
        set.drop("income_cat",axis=1,inplace=True)
#%%
house_train=train_set.copy()
#%%
# 地理分布图
%matplotlib inline
house_train.plot(kind="scatter",x='longitude',y='latitude',alpha=0.1)

#%%
# 人口、房价分布
house_train.plot(kind="scatter",x='longitude',y='latitude',alpha=0.4,s=house_train['population']/100,
label="population",c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
#%%
# train_test_split
#%%
# 相关系数
corr_matrix=house_train.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
from pandas.plotting import scatter_matrix
cols=['longitude','latitude']
scatter_matrix(house[cols],figsize=(12,8))

#%%
# 属性组合
house_train["rooms_per_household"] = house_train["total_rooms"]/house_train["households"]
house_train["bedrooms_per_room"] = house_train["total_bedrooms"]/house_train["total_rooms"]
house_train["population_per_household"]=house_train["population"]/house_train["households"]

corr_matrix=house_train.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
# 数据准备
housing=house_train.drop('median_house_value',axis=1)
labels=house_train['median_house_value'].copy()

#%%
# 缺失值处理
# print(housing.info())
# housing=housing.dropna(subset=['total_bedrooms'])

# housing.info()


#%%
median = housing["total_bedrooms"].median()
housing["total_bedrooms"]=housing["total_bedrooms"].fillna(median)
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing.info()
#%%
# Imputer
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
housing_num=housing.drop('ocean_proximity',axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
x=imputer.transform(housing_num)
x.shape
#%%
housing_tr=pd.DataFrame(x,columns=housing_num.columns)
housing_tr.describe()
#%%
# 文本类别数据转换
#有问题 该转换器只能用来转换标签 在这里使用 LabelEncoder  没有出错的原因是该数据只有一列
#文本特征值，在有多个文本特征列的时候就会出错
from sklearn.preprocessing import LabelEncoder
house_cat=housing['ocean_proximity']
encoder=LabelEncoder()
house_cat_encoder=encoder.fit_transform(house_cat)
encoder.classes_


#%%
# 正确转换方式
housing_cat_encoded, housing_categories = house_cat.factorize()

#%%
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
house_onehot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
house_onehot.toarray()

#%%
# 合并
from data  import CategoricalEncoder
encoder=CategoricalEncoder()
house_onehot=encoder.fit_transform(house_cat.values.reshape(-1,1))
house_onehot.toarray()

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

pipline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('adder',CombinedAttributesAdder()),
    ('standarize',StandardScaler())

])
housing_num_transform=pipline.fit_transform(housing_num)
housing_num_transform
#%%
from sklearn.pipeline import FeatureUnion
from data  import CategoricalEncoder
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder(a)),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', CategoricalEncoder()),
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
house_res=full_pipeline.fit_transform(housing)
house_res=house_res.toarray()
#%%
# 线性回归
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(house_res,labels)

#%%
# 线性回归
data_pre=housing[:5]
label_pre=labels.iloc[:5].values
print(data_pre)
print(label_pre)
some_data_pre=full_pipeline.transform(data_pre)
print("Predictions:\t", model.predict(some_data_pre))
print(label_pre)

#%%
from sklearn.metrics import mean_absolute_error
housing_pre=full_pipeline.transform(housing)
predict=model.predict(housing_pre)
mse=mean_absolute_error(labels,predict)
print(mse)
np.sqrt(mse)

#%%
# 决策树
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(housing_pre,labels)
predict=tree.predict(housing_pre)
mse=mean_absolute_error(labels,predict)
print(mse)
np.sqrt(mse)

#%%
# k折交叉验证
from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree,housing_pre,labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores=np.sqrt(-scores)
print(scores)
print(rmse_scores)
#%%
print(scores.mean())
print(rmse_scores.mean())
#%%
# 随机森林
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(housing_pre,labels)
# predict=forest.predict(housing_pre)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(forest,housing_pre,labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores=np.sqrt(-scores)
print(scores)
print(rmse_scores)
#%%
print(scores.mean())
print(rmse_scores.mean())

#%%
# 保存模型
from sklearn.externals import joblib
joblib.dump(forest,'forest.pkl')
# 载入模型
# my_model_loaded = joblib.load("my_model.pkl")

#%%
# 模型微调
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
gridsearch=GridSearchCV(forest,param_grid,scoring='neg_mean_squared_error',cv=5)
gridsearch.fit(housing_pre,labels)


#%%
print(gridsearch.best_params_)
cvres=gridsearch.cv_results_
for meanscore,param in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-meanscore),param)

#%%
# 每个属性对于做出准确预测的相对重要性：
gridsearch.best_estimator_.feature_importances_

#%%
# 将重要性分数和属性名放到一起：


#%%
# 测试
from sklearn.metrics import mean_absolute_error
final_model=gridsearch.best_estimator_
X_test=test_set.drop('median_house_value',axis=1)
X_test["rooms_per_household"] = X_test["total_rooms"]/X_test["households"]
X_test["bedrooms_per_room"] = X_test["total_bedrooms"]/X_test["total_rooms"]
X_test["population_per_household"]=X_test["population"]/X_test["households"]

Y_test=test_set['median_house_value'].copy()
X_test_pre=full_pipeline.transform(X_test)
final_predict=final_model.predict(X_test_pre)
final_mse = mean_absolute_error(Y_test, final_predict)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6
print(final_mse,final_rmse)

#%%

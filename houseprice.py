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
print(housing.info())
housing=housing.dropna(subset=['total_bedrooms'])

housing.info()

#%%
housing.drop('total_bedrooms',axis=1,inplace=True)
housing.info()


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
from sklearn.preprocessing  import CategoricalEncoder
encoder=CategoricalEncoder()
house_onehot=encoder.fit_transform(housing_cat.reshape(-1,1))
house_onehot

#%%

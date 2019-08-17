#%%
import os
import pandas as pd
path=os.getcwd()
money=pd.read_csv(path+"\data\data.csv")
satisfy=pd.read_csv(path+"\data\data2.csv")
money.describe()
# print("satisfy")
# satisfy.describe()
#%%
data=pd.merge(money,satisfy,on='Country')
#%%
def convert_currency(var):
    new_value = var.replace(",","").replace("$","")
    return float(new_value)

data['2015']=data['2015'].apply(convert_currency)
data
#%%
import matplotlib.pyplot as plt

x=data['2015']
y=data['satisfy']
plt.figure(figsize=(30,20))
plt.scatter(x,y)
ax = plt.gca()                                            # get current axis 获得坐标轴对象
ax.set_ylim([0, 10])
plt.show()

#%%
from sklearn.linear_model import LinearRegression
import numpy as np
print(len(x))
x=np.array(x)
y=np.array(y)
model=LinearRegression()
model.fit(x,y)
model.predict([[23456]])

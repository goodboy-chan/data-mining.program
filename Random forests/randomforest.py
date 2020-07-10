import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor #随机森林
from sklearn.model_selection import train_test_split #随机划分训练子集与测试子集
from statsmodels.stats.outliers_influence import variance_inflation_factor #多重共线性方差膨胀因子

#读取数据
train_data0=pd.read_table('./data/zhengqi_train.txt',sep='\t')
test_data0=pd.read_table('./data/zhengqi_test.txt',sep='\t')

#训练数据总览
train_data0.info()

# 没有空缺值，由于指标的含义暂不清晰，异常值不进行处理

train_data0.describe()

#训练数据分布情况
plt.figure(figsize=(18, 18))

for column_index, column in enumerate(train_data0.columns):
    plt.subplot(10, 4, column_index + 1)
    g=sns.kdeplot(train_data0[column])
    g.set_xlabel(column)
    g.set_ylabel('Frequency')

#特征相关性
plt.figure(figsize=(20, 16))
colnm = train_data0.columns.tolist()
mcorr = train_data0[colnm].corr(method="spearman")
mask = np.zeros_like(mcorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
plt.show()

mcorr=mcorr.abs()
numerical_corr=mcorr[mcorr['target']>0.1]['target']
print(numerical_corr.sort_values(ascending=False))
index0 = numerical_corr.sort_values(ascending=False).index
print(train_data0[index0].corr('spearman'))

# 自变量与目标变量相关性绝对值0.1以上的有：
# V0,V1,V2,V3,V4,V5,V6,V7,V8,V10,V11,V12,V13,V15,V16,V18,V19,V20,V22,V23,V24,V27,V29,V30,V31,V36,V37
# 自变量相关性0.8以上的有：
# V0: V1 V8 ;
# V1: V8 V27 V31;
# V4: V12;
# V5: V11;
# V6: V7;
# V8: V27 V31 ;
# V10: V36;
# V15: V29;
# V23: V35;
# 于是，选取特征自变量与目标变量相关性绝对值0.1以上并且剔除高度相关性的自变量。

train_data1=train_data0[['V1','V2','V3','V4','V5','V6','V10','V13','V15',
'V16','V18','V19','V20','V22','V23','V24','V30','V37','target']]
train_data1.head()

#多重共线性
new_numerical=['V1','V2','V3','V4','V5','V6','V10','V13','V15',
'V16','V18','V19','V20','V22','V23','V24','V30','V37']
X=np.matrix(train_data1[new_numerical])
VIF_list=[variance_inflation_factor(X, i) for i in range(X.shape[1])]
VIF_list

# 多重共线性不明显，暂不需要进一步降维处理。为了降低个别特征较大波动，有可能造成不同特征权重系数变化过大，将数据进行z-score标准化。

train0=train_data1.iloc[:,0:-1]
test0=test_data0[['V1','V2','V3','V4','V5','V6','V10','V13','V15',
'V16','V18','V19','V20','V22','V23','V24','V30','V37']]
target=train_data1.iloc[:,-1]
train=(train0-np.mean(train0,axis=0))/np.std(train0,axis=0)
test=(test0-np.mean(test0,axis=0))/np.std(test0,axis=0)

# 预测模型探索

train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.2,random_state=0)
m=RandomForestRegressor()
m.fit(train_data, train_target)
score=mean_squared_error(test_target,m.predict(test_data))
print(score)

# 多次重复，查看模型预测结果的稳定性

model_accuracies = []

for repetition in range(100):
    train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.2,random_state=0)

    m=RandomForestRegressor()
    m.fit(train_data, train_target)
    score=mean_squared_error(test_target,m.predict(test_data))
    model_accuracies.append(score)

sns.distplot(model_accuracies)

# 模型参数调优

param_grid = {'n_estimators':[1,5,10,25,50,100],
'max_features':('auto','sqrt','log2')}
m = GridSearchCV(RandomForestRegressor(),param_grid)
m=m.fit(train_data,train_target)
score=mean_squared_error(test_target,m.predict(test_data))
print(score)
print(m.best_score_)
print(m.best_params_)

m = RandomForestRegressor(n_estimators=100,max_features='log2')
m.fit(train_data, train_target)
predict = m.predict(test)
np.savetxt('predict.txt',predict)

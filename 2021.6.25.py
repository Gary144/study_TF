'''
这是一个泰塔尼克号上的生存预测数据
作者:Gary
时间:6.25
'''

filepath=r"C:\Users\Administrator\PycharmProjects\data\titanic3.xls" #本地数据文件路径

import numpy as np
import pandas as pd
from sklearn import preprocessing

all_df = pd.read_excel(filepath)

cols = ['survived',
        'name',
        'pclass',
        'sex',
        'age',
        'sibsp',
        'parch',
        'fare',
        'embarked']

all_df = all_df[cols]

print(all_df)

def ProssingData(raw_df):
    df = all_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    x_one_hot_df = pd.get_dummies(data=df, columns=["embarked"])

    ndarry = x_one_hot_df.values
    Label = ndarry[:, 0]
    Features = ndarry[:, 1:]

    minmax_sacle = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_sacle.fit_transform(Features)
    return scaledFeatures,Label


# 将数据集随机分为训练集和测试集

msk = np.random.rand(len(all_df))<0.8
train_df = all_df[msk]
test_df = all_df[~msk]

train_Features,train_Label = ProssingData(train_df)
test_Features,test_Label = ProssingData(test_df)

# 建立模型
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(
    Dense(units=40,input_dim=9,kernel_initializer='uniform',activation='relu')
)

model.add(Dense(
    units=30,kernel_initializer='uniform',activation='relu'
))

model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


# 开始训练

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit(x=train_Features,
                          y=train_Label,
                          validation_split=0.0001,
                          epochs=1000,
                          batch_size=30,
                          verbose=2)

scores = model.evaluate(x=test_Features,y=test_Label)
print(scores[1])

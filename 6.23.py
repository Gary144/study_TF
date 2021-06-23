import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)

from keras.datasets import mnist
(x_train_image, y_train_label), (x_test_image, y_test_label) =mnist.load_data()

print('train_data=', len(x_train_image))
print('test_data=', len(x_test_image))

# 训练数据是由images和labels组成的
print('x_train_image:', x_train_image.shape)
print('x_test_image:', x_test_image.shape)

import matplotlib.pyplot as plt

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()

# 可以尝试去查看相关的数据
plot_image(x_train_image[0])

y_train_label[0]

# 查看多项真实数据和预测结果

def plot_images_labels_prediction(images, labels,
                                  prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25 : num = 25
    for i in range (0, num):
        ax=plt.subplot(5, 5, i+1)
        ax.imshow(images[idx], cmap='binary')
        title="label="+str(labels[idx])
        if len(prediction) > 0:
            title += ",predict="+str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()

# 用于查看训练数据
plot_images_labels_prediction(x_train_image, y_train_label, [], 0, 10)

print('x_test_image:',x_test_image.shape)
print('y_test_label:',y_test_label.shape)

# 用于查看测试数据
plot_images_labels_prediction(x_test_image, y_test_label, [], 0, 10)

# features数据预处理
# 先查看每一个数字图像的shape
print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

x_train = x_train_image.reshape(60000, 784).astype('float32')
x_test = x_test_image.reshape(10000, 784).astype('float32')

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)

x_train_normalize = x_train / 255
x_test_normalize = x_test / 255

# 查看label里面的内容
y_train_label[:5]

# 将label里面的内容转化为独热编码,独热编码的作用是避免数值对于计算机判断产生影响
y_train_onehot = np_utils.to_categorical(y_train_label)
y_test_onehot = np_utils.to_categorical(y_test_label)

# 到此为止完成了所有的数据预处理

# 下面搭建的是多层感知机的识别模型
'''
只含有线性层的感知分类机器
搭建下列神经元：
输入层：784个神经元
隐藏层：256个神经元
输出层：10个神经元
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(
    Dense(units=256,
          input_dim=784,
          kernel_initializer='normal',
          activation='relu')
)

model.add(
   Dropout(0.5)
)

model.add(
    Dense(units=10,
          kernel_initializer='normal',
          activation='softmax')
)

print(model.summary())

# 在训练之前,必须对模型进行初始化

'''
loss:损失函数
optimizer:优化器
metrics:评估模型的方式
'''

model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

'''
x=x_train_normalize:features数字图像的特征
y=y_train_onehot:label数字图像真实的值
validation_split=0.2:训练之前keras会自动将数据分成两个part,一部分用于训练,一个part用于验证
epochs=10:训练迭代的次数
batch_size=200:每一批次训练的数据量
verbose=2:显示训练的过程
'''

train_history = model.fit(x=x_train_normalize,
                          y=y_train_onehot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=2
                          )

# 下面这段代码有点小问题,待debug
'''
# 建立show_train_history显示训练过程

import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

acc ='acc'
validation = 'val_acc'

show_train_history(train_history,acc,validation)
'''

scores = model.evaluate(x_test_normalize, y_test_onehot)
print( )
print('accuracy=',scores[1])

prediction = model.predict(x_test)
print(prediction)



# 下面搭建的是卷积神经网络CNN模型
'''
CNN与多层感知机的不同之处在于
数据的输入:对于多层感知机来说,神经元为分散线性分布,最终组合成非线性效果:因此对于多层感知机,输入为一维
         对于卷积神经网络来说,神经元模拟了大脑的视觉模式进行学习,最终组合形成非线性效果:因此对于卷积神经网络来说,输入为二维
模型的架构:架构有明显区别
内部的机制:类似
'''

# 数据的预处理
import numpy as np
np.random.seed(10)

x_train4D = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test4D = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train4D_normalized = x_train4D / 255
x_test4D_normalized = x_test4D / 255

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model1 = Sequential()

model1.add(
    Conv2D(filters=16,
           kernel_size=(5, 5),
           padding='same',
           input_shape=(28, 28, 1),
           activation='relu')
)

model1.add(
    MaxPooling2D(pool_size=(2, 2))
)

model1.add(
    Conv2D(
        filters=36,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )
)

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Dropout(0.25))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dropout(0.5))

model1.add(Dense(
    10,activation='softmax'
))

print(model1.summary())

model1.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

train_history=model1.fit(x=x_train4D_normalized,
                         y=y_train_onehot,
                         validation_split=0.2,
                         epochs=10,
                         batch_size=300,
                         verbose=2)





















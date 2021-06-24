'''
CIFAR10 数字图像识别实战
作者:gary
'''

from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()

print('train:',len(x_img_train))
print('test:',len(x_img_test))

print(x_img_train.shape)
print(x_img_test[0])
print(y_label_train.shape)

# 定义每一个数字代表的图像类别的名称
label_dict={
            0:"airplane",
            1:"automobile",
            2:"bird",
            3:"cat",
            4:"deer",
            5:"dog",
            6:"frog",
            7:"horse",
            8:"ship",
            9:"truck"
            }

import matplotlib.pyplot as plt
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:num=25
    for i in range(num):
        ax=plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]

        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()

plot_image_labels_prediction(x_img_train,y_label_train,[],0)

x_img_train_normal=x_img_train/255.0
x_img_test_normal=x_img_test/255.0

from keras.utils import np_utils
y_label_train_onehot=np_utils.to_categorical(y_label_train)
y_label_test_onehot=np_utils.to_categorical(y_label_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D

model=Sequential()
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(32,32,3),
    activation='relu',
    padding='same'
))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu',
    padding='same'
))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024,activation='relu'))
model.add(Dense(10,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

try:
    model.load_weights("SaveModel/cifarCnnModel.h5")
    print('Model load successfully')
except:
    print('Model load failed')
    
train_history=model.fit(x_img_train_normal,y_label_train_onehot,validation_split=0.2,epochs=10,batch_size=128,verbose=1)
model.save_weights("SaveModel/cifarCnnModel.h5")
print("Saved model to disk")

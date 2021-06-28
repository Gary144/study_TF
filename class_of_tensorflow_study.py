import tensorflow as tf
import tensorflow.keras as keras

class LeNet(keras.Model):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv2d_1 = keras.layers.Conv2D(filters=6,
                                            kernel_size=(3,3),
                                            activation='relu',
                                            input_shape=(32,32,1))
        self.aveage_pool = keras.layers.AveragePooling2D()
        self.conv2d_2 = keras.layers.Conv2D(filters=16,
                                            kernel_size=(3,3),
                                            activation='relu')
        self.flatten = keras.layers.Flatten()
        self.fc_1 = keras.layers.Dense(120,activation='relu')
        self.fc_2 = keras.layers.Dense(84,activation='relu')
        self.out = keras.layers.Dense(10,activation='softmax')

    def call(self,input):
        x = self.conv2d_1(input)
        x = self.aveage_pool(x)
        x = self.conv2d_2(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.out(x)

        return self.out(x)


import tensorflow as tf
import matplotlib.pyplot as plt

(mnist_images, mnist_labels), _=tf.keras.datasets.mnist.load_data()

dataset=tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[...,tf.newaxis]/255,tf.float32),
         tf.cast(mnist_labels,tf.int64))
    )
dataset=dataset.shuffle(1000).batch(32)

mnist_model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3],activation='relu',input_shape=(None,None,1)),
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])


    # For example:
optimizer=tf.keras.optimizers.Adam()
loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history=[]

def train_step(images,labels):
    with tf.GradientTape() as tape:
        logits=mnist_model(images, training=True)

        #Add assert to check the shape of the output:
        tf.debugging.assert_equal(logits.shape,(32,10))

        loss_value=loss_object(labels,logits)

    loss_history.append(loss_value.numpy().mean())
    grads=tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train(Epochs):
    for Epoch in range (Epochs):
        for (batch,(images,labels)) in enumerate(dataset):
            train_step(images,labels)
        print('Epoch{} finished'.format(Epoch))



# 没有训练，在EE中可以调用模型检查输出

if __name__=='__main__':
    train(Epochs=3)
    plt.plot(loss_history)
    plt.xlabel('Batch #:')
    plt.ylabel('loss[Entropy]')
    plt.show()

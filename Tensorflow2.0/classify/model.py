import tensorflow as tf
from tensorflow.keras import layers
def myModel(input_shape,load_model = True):
    if load_model == True:
        myVGG = tf.keras.applications.VGG16(weights = 'imagenet',include_top=False)
        myVGG.trainable =False
    else:
        myVGG = tf.keras.Sequential()

        myVGG.add(layers.Conv2D(64,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(64,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.MaxPooling2D())

        myVGG.add(layers.Conv2D(128,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(128,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.MaxPooling2D())

        myVGG.add(layers.Conv2D(256,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(256,(3,3),padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(256, (1, 1), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.MaxPooling2D())

        myVGG.add(layers.Conv2D(512, (3, 3), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(512, (3, 3), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(512, (1, 1), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.MaxPooling2D())

        myVGG.add(layers.Conv2D(512, (3, 3), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(512, (3, 3), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))

        myVGG.add(layers.Conv2D(512, (1, 1), padding='same'))
        myVGG.add(layers.BatchNormalization())
        myVGG.add(layers.Activation('relu'))



    inputs = tf.keras.layers.Input(shape=input_shape)
    x = myVGG(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1000, activation='relu')(x)
    outputs = layers.Dense(20,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs,
                                  outputs=outputs)
    return model
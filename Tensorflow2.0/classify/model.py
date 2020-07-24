import tensorflow as tf
from tensorflow.keras import layers
def myModel(input_shape,load_model = True):
    if load_model == True:
        VGG16 = tf.keras.applications.VGG16(weights = 'imagenet',include_top=False)
        VGG16.trainable =False
    else:
        pass
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = VGG16(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs,
                                  outputs=outputs)
    return model
import tensorflow as tf
from tensorflow.keras import layers


input_shape = (256,256,3)
def model(input_shape,class_num):
    inputs = layers.Input(shape = input_shape)
    x = layers.Conv2D(64,3,padding='same',activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x) #shape = (256,256,64)

    x1 = layers.MaxPooling2D()(x) #shape = (128,128,64)
    x1 = layers.Conv2D(128,3,padding='same',activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(128,3,padding='same',activation='relu')(x1)
    x1 = layers.BatchNormalization()(x1) #shape = (128,128,128)

    x2 = layers.MaxPooling2D()(x1) #shape = (64,64,128)
    x2 = layers.Conv2D(256,3,padding='same',activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv2D(256,3,padding='same',activation='relu')(x2)
    x2 = layers.BatchNormalization()(x2) #shape = (64,64,256)

    x3 = layers.MaxPooling2D()(x2) #shape = (32,32,256)
    x3 = layers.Conv2D(512,3,padding='same',activation='relu')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Conv2D(512,3,padding='same',activation='relu')(x3)
    x3 = layers.BatchNormalization()(x3) #shape = (32,32,512)

    x4 = layers.MaxPooling2D()(x3) #shape = (16,16,512)
    x4 = layers.Conv2D(1024,3,padding='same',activation='relu')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Conv2D(1024,3,padding='same',activation='relu')(x4)
    x4 = layers.BatchNormalization()(x4) #shape = (16,16,1024)

    #上采样
    x5 = layers.Conv2DTranspose(512,2,strides=2,padding='same',activation='relu')(x4)
    x5 = layers.BatchNormalization()(x5) #shape = (32,32,512)

    x6 = tf.concat([x3,x5],axis = -1) #shape = (32,32,1024)
    x6 = layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.Conv2D(512, 3, padding='same', activation='relu')(x6)
    x6 = layers.BatchNormalization()(x6) #shape = (32,32,512)
    # 上采样
    x7 = layers.Conv2DTranspose(256,2,strides=2,padding='same',activation='relu')(x6)
    x7 = layers.BatchNormalization()(x7) #shape = (64,64,256)

    x8 = tf.concat([x2, x7], axis=-1)  # shape = (64,64,512)
    x8 = layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.Conv2D(256, 3, padding='same', activation='relu')(x8)
    x8 = layers.BatchNormalization()(x8) #shape = (64,64,256)
    # 上采样
    x9 = layers.Conv2DTranspose(128,2,strides=2,padding='same',activation='relu')(x8)
    x9 = layers.BatchNormalization()(x9) #shape = (128,128,128)

    x10 = tf.concat([x1, x9], axis=-1)  # shape = (128,128,256)
    x10 = layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = layers.BatchNormalization()(x10)
    x10 = layers.Conv2D(128, 3, padding='same', activation='relu')(x10)
    x10 = layers.BatchNormalization()(x10) #shape = (128,128,128)
    # 上采样
    x11 = layers.Conv2DTranspose(64,2,strides=2,padding='same',activation='relu')(x10)
    x11 = layers.BatchNormalization()(x11) #shape = (256,256,64)

    x12 = tf.concat([x, x11], axis=-1)  # shape = (256,256,128)
    x12 = layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = layers.BatchNormalization()(x12)
    x12 = layers.Conv2D(64, 3, padding='same', activation='relu')(x12)
    x12 = layers.BatchNormalization()(x12) #shape = (128,128,64)

    outputs = layers.Conv2D(class_num, 1, padding='same', activation='softmax')(x12)

    return tf.keras.Model(inputs = inputs,outputs = outputs)

mymodel = model(input_shape,34)
tf.keras.utils.plot_model(mymodel)
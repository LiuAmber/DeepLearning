from LoadData import LoadData
from UNetModel import model
from metrics import MeanIoU
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

PATH = "E:/dataset/日月光华-tensorflow资料/数据集/UNET语义分割/城市街景数据集"

def train(path):
    BATCH_SIZE = 1
    input_shape = (256,256,3)
    class_num = 34
    data = LoadData(path,BATCH_SIZE)
    train_data = data.train_data()
    test_data = data.test_data()
    train_count = data.train_len
    test_count = data.test_len
    STEPS_PER_EPOCH = train_count // BATCH_SIZE
    VALIDATION_STEPS = test_count // BATCH_SIZE
    myModel = model(input_shape,class_num)
    myModel.compile(optimizer='adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics=['acc',MeanIoU(num_classes=class_num)])
    EPOCHS = 60
    history = myModel.fit(train_data,
                        epochs = EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=test_data,
                          validation_steps=VALIDATION_STEPS)


train(path=PATH)
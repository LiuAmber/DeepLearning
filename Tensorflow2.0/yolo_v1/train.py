import tensorflow as tf
from LoadVocData import LoadVocData
from Loss import yolov1_loss
from Model import yolov1_model
import os
def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PATH = 'C:\\Users\\10372\\Desktop\\VOC\\VOCdevkit\\VOC2012'
    train_count = 5000
    BATCH_SIZE = 16
    train_data = LoadVocData(PATH).train_data(BATCH_SIZE)
    steps_per_epoch = train_count // BATCH_SIZE
    num_epochs = 30
    model = yolov1_model()
    model.compile(optimizer='adam',
                  loss = yolov1_loss,
                  metrics=['acc'])
    model.fit(train_data,epochs = num_epochs,
              steps_per_epoch=steps_per_epoch)

train()
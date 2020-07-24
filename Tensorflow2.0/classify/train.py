from classify.LoadImage import LoadImage
from classify.model import myModel
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    path = 'C:\\Users\\10372\\Desktop\\image'
    batch_size = 16
    shape = (224, 224, 3)
    image_data = LoadImage(path,batch_size)
    train_ds = image_data.train_data()
    test_ds = image_data.test_data()
    train_count = image_data.train_count
    test_count = image_data.test_count
    model = myModel(shape)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
             loss = 'categorical_crossentropy',
             metrics=['acc'])
    EPOCHS = 20
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        steps_per_epoch=train_count,
                        validation_data=test_ds,
                        validation_steps=test_count)

train()

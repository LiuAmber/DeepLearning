import tensorflow as tf

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self,y_true,y_pred,sample_weight = None):
        y_pred = tf.argmax(y_pred,axis=-1)
        return super().__call__(y_true,y_pred,sample_weight = sample_weight)
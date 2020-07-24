import tensorflow as tf
import numpy as np
import glob

class LoadImage:
    def __init__(self,path,batch_size):
        self.path = path
        self.kinds = {}
        self.batch_size = batch_size
        self.train_count = 0
        self.test_count = 0
    def __get_kinds__(self):
        kinds_path = glob.glob(self.path+"\\train\\*")
        i = 0
        for kind_path in kinds_path:
            self.kinds[kind_path.split("\\")[-1]] = i
            i += 1

    def __kind_to_number__(self,kinds):
        kind_numbered = []
        for kind in kinds:

            temp = np.zeros((len(self.kinds),1))
            temp[self.kinds[kind]] = 1
            kind_numbered.append(temp)
        return kind_numbered


    def __load_image__(self,path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.cast(img,tf.float32)
        img = img/127.5-1   #image normalization
        return img

    def train_data(self):
        self.__get_kinds__()
        img_path = glob.glob(self.path+"\\train\\"+"*"+"\\*.jpg")
        self.train_count = len(img_path)
        train_label = [path.split("\\")[-2] for path in img_path]
        train_label_indexed = self.__kind_to_number__(train_label)
        train_label_ds = tf.data.Dataset.from_tensor_slices(train_label_indexed)
        train_img_ds = tf.data.Dataset.from_tensor_slices(img_path)
        train_img_ds = train_img_ds.map(self.__load_image__)
        train_ds = tf.data.Dataset.zip((train_img_ds,train_label_ds))
        train_ds = train_ds.repeat().shuffle(self.train_count//100).batch(self.batch_size)
        return train_ds

    def test_data(self):
        img_path =  glob.glob(self.path+"\\test\\"+"*"+"\\*.jpg")
        self.test_count = len(img_path)
        test_label = [path.split("\\")[-2] for path in img_path]
        test_label_indexed = self.__kind_to_number__(test_label)
        test_label_ds = tf.data.Dataset.from_tensor_slices(test_label_indexed)
        test_img_ds = tf.data.Dataset.from_tensor_slices(img_path)
        test_img_ds = test_img_ds.map(self.__load_image__)
        test_ds = tf.data.Dataset.zip((test_img_ds,test_label_ds))
        test_ds = test_ds.repeat().batch(self.batch_size)
        return test_ds



path = 'C:\\Users\\10372\\Desktop\\image'
batch_size = 16
LoadImage(path,batch_size).train_data()



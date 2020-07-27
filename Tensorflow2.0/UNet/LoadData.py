import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

PATH = "E:/dataset/日月光华-tensorflow资料/数据集/UNET语义分割/城市街景数据集"

class LoadData:
    def __init__(self,path,batch_size):
        self.path = path
        self.crop_shape = (256,256,4)
        self.resize_shape = (280,280)
        self.batch_size = batch_size
        self.train_len = 0
        self.test_len = 0
        self.buffer_size = 0

    def __read_png_img__(self,path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img,channels=3)
        return img

    def __read_png_label__(self,path):
        label = tf.io.read_file(path)
        label = tf.image.decode_png(label,channels=1)
        return label

    def __crop_img__(self,img,mask):
        concat_img = tf.concat([img,mask],axis=-1)
        resized_img = tf.image.resize(concat_img,self.resize_shape,
                                      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        crop_img = tf.image.random_crop(resized_img,self.crop_shape)
        return crop_img[:,:,:3],crop_img[:,:,3:]

    def __img_normal__(self,img,mask):
        img = tf.cast(img,tf.float32)/127.-1
        mask = tf.cast(mask,tf.int32)
        return img,mask

    def __load__image_train__(self,img_path,label_path):
        img = self.__read_png_img__(img_path)
        mask = self.__read_png_label__(label_path)
        img,mask = self.__crop_img__(img,mask)
        if tf.random.uniform(())>0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        img,mask = self.__img_normal__(img,mask)
        return img,mask

    def __load_image_test__(self,img_path,label_path):
        img = self.__read_png_img__(img_path)
        mask = self.__read_png_label__(label_path)

        img = tf.image.resize(img,(self.crop_shape[0],self.crop_shape[1]))
        mask = tf.image.resize(mask,(self.crop_shape[0],self.crop_shape[1]))

        img,mask = self.__img_normal__(img,mask)
        return img,mask

    def train_data(self):
        img_path = glob.glob(self.path+'/images/train/*/*.png')
        label_path = glob.glob(self.path+'/gtFine/train/*/*_gtFine_labelIds.png')
        self.train_len = len(img_path)
        self.buffer_size = self.train_len//10
        index = np.random.permutation(self.train_len)
        img_path = np.array(img_path)[index]
        label_path = np.array(label_path)[index]
        auto= tf.data.experimental.AUTOTUNE
        train_dataset = tf.data.Dataset.from_tensor_slices((img_path,label_path))
        train_dataset = train_dataset.map(self.__load__image_train__,num_parallel_calls = auto)
        train_dataset = train_dataset.cache().repeat().shuffle(self.buffer_size).batch(self.batch_size).prefetch(auto)
        return train_dataset

    def test_data(self):
        img_path = glob.glob(self.path+'/images/val/*/*.png')
        label_path = glob.glob(self.path+'/gtFine/val/*/*_gtFine_labelIds.png')

        self.test_len = len(img_path)
        index = np.random.permutation(self.test_len)
        img_path = np.array(img_path)[index]
        label_path = np.array(label_path)[index]
        auto= tf.data.experimental.AUTOTUNE
        test_dataset = tf.data.Dataset.from_tensor_slices((img_path,label_path))
        test_dataset = test_dataset.map(self.__load_image_test__,num_parallel_calls = auto)
        test_dataset = test_dataset.cache().batch(self.batch_size).prefetch(auto)
        return test_dataset


    def test(self):
        data = self.train_data()
        index = 0
        for i, m in data.take(1):
            plt.subplot(1,2,1)
            plt.imshow((i[index].numpy()+1)/2)
            plt.subplot(1,2,2)
            plt.imshow(np.squeeze(m[index].numpy()))
            print(np.unique(m[index].numpy()))
            plt.show()

# LoadData(PATH,32).test()
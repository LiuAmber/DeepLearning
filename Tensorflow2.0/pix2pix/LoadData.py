import tensorflow as tf
import os
import matplotlib.pyplot as plt


class LoadData:
    def __init__(self):
        self.URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
        self.BUFFER_SIZE = 400
        self.BATCH_SIZE = 1
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
        self.PATH = ''
        self.__get_path__()

    def __get_path__(self):
        path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                              origin=self.URL,
                                              extract=True)

        self.PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

    def __load__(self,image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]

        w = w // 2
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def __resize__(self,input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    def __random_crop__(self,input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

        return cropped_image[0], cropped_image[1]

    def __normalize__(self,input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    @tf.function()
    def __random_jitter__(self,input_image, real_image):
        # resizing to 286 x 286 x 3
        input_image, real_image = self.__resize__(input_image, real_image, 286, 286)

        # randomly cropping to 256 x 256 x 3
        input_image, real_image = self.__random_crop__(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def __load_image_train__(self,image_file):
        input_image, real_image = self.__load__(image_file)
        input_image, real_image = self.__random_jitter__(input_image, real_image)
        input_image, real_image = self.__normalize__(input_image, real_image)

        return input_image, real_image

    def __load_image_test__(self,image_file):
        input_image, real_image = self.__load__(image_file)
        input_image, real_image = self.__resize__(input_image, real_image,
                                         self.IMG_HEIGHT, self.IMG_WIDTH)
        input_image, real_image = self.__normalize__(input_image, real_image)

        return input_image, real_image

    def train_data(self):
        train_dataset = tf.data.Dataset.list_files(self.PATH + 'train/*.jpg')
        train_dataset = train_dataset.map(self.__load_image_train__,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)
        return train_dataset

    def test_data(self):
        test_dataset = tf.data.Dataset.list_files(self.PATH + 'test/*.jpg')
        test_dataset = test_dataset.map(self.__load_image_test__)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)
        return test_dataset

# data = LoadData()
# PATH = data.PATH
#
# inp, re = data.__load__(PATH+'train/100.jpg')
# # casting to int for matplotlib to show the image
# plt.figure()
# plt.imshow(inp/255.0)
# plt.figure()
# plt.imshow(re/255.0)
# plt.show()



import tensorflow as tf
import matplotlib.pyplot as plt
from lxml import etree
import numpy as np
import glob
# import os
from matplotlib.patches import Rectangle
from PIL import Image

class LoadVOCdata:
    def __init__(self, path,S,B,C,image_resize):
        self.path = path
        self.images_path = glob.glob(path + "\\JPEGImages\\*.jpg")
        self.xmls_path = (path + "\\Annotations\\*.xml")

        self.S = S
        self.C = C
        self.B = B
        self.image_resize = image_resize

        self.train_images_name = []
        self.val_images_name = []
        self.test_images_name = []

        self.train_images_path = []
        self.val_images_path = []
        self.test_images_path = []

        self.train_labels_path = []
        self.val_labels_path = []
        self.test_labels_path = []

        self.kind = []

    def __getTrainPath__(self):
        train_names = open(self.path + "\\ImageSets\\Main\\train.txt", 'r')
        for name in train_names:
            self.train_images_name.append(name.split('\n')[0])
            self.train_images_path.append(self.path + "\\JPEGImages\\" + name.split('\n')[0] + ".jpg")
            self.train_labels_path.append(self.path + "\\Annotations\\" + name.split('\n')[0] + ".xml")

    def __getValidationPath__(self):
        val_names = open(self.path + "\\ImageSets\\Main\\trainval.txt", 'r')
        for name in val_names:
            self.val_images_name.append(name.split('\n')[0])
            self.val_images_path.append(self.path + "\\JPEGImages\\" + name.split('\n')[0] + ".jpg")
            self.val_labels_path.append(self.path + "\\Annotations\\" + name.split('\n')[0] + ".xml")

    def __getTestPath__(self):
        test_names = open(self.path + "\\ImageSets\\Main\\val.txt", 'r')
        for name in test_names:
            self.test_images_name.append(name.split('\n')[0])
            self.test_images_path.append(self.path + "\\JPEGImages\\" + name.split('\n')[0] + ".jpg")
            self.test_labels_path.append(self.path + "\\Annotations\\" + name.split('\n')[0] + ".xml")

    def __toLabels__(self,path):
        xml = open('{}'.format(path)).read()
        sel = etree.HTML(xml)
        height = int(sel.xpath('//size/height/text()')[0])
        width = int(sel.xpath('//size/width/text()')[0])
        xmins = np.array(sel.xpath('//object/bndbox/xmin/text()')).astype(np.int)
        ymins = np.array(sel.xpath('//object/bndbox/ymin/text()')).astype(np.int)
        xmaxs = np.array(sel.xpath('//object/bndbox/xmax/text()')).astype(np.int)
        ymaxs = np.array(sel.xpath('//object/bndbox/ymax/text()')).astype(np.int)
        classes = sel.xpath('//object/name/text()')
        return [xmins / width, ymins / height, xmaxs / width, ymaxs / height, classes]

    def __kindToNumber__(self,labels):
        output1, output2, output3, output4, output5 = list(zip(*labels))
        for i in range(len(output5)):
            for j in range(len(output5[i])):
                if output5[i][j] not in self.kind:
                    self.kind.append(output5[i][j])
        output5 = np.array(output5)
        for i in range(len(output5)):
            for j in range(len(output5[i])):
                for k in range(len(self.kind)):
                    if output5[i][j] == self.kind[k]:
                        output5[i][j] = k
                        output5[i][j] = int(output5[i][j])
            output5[i] = np.array(output5[i])
        return [output1, output2, output3, output4, output5]

    def __labelsNormalization__(self,labels):
        print(len(labels))
        xmins_n,ymins_n,xmaxs_n,ymaxs_n,class_num = labels[0],labels[1],labels[2],labels[3],labels[4]
        label_tensor = np.zeros((len(xmins_n),self.S,self.S,self.B*5+self.C))
        for i in range(len(xmins_n)):
          for j in range(len(xmins_n[i])):
            S_i = int((xmaxs_n[i][j]+ xmins_n[i][j])/2*self.S)
            S_j = int((ymaxs_n[i][j]+ ymins_n[i][j])/2*self.S)
            label_tensor[i][S_i][S_j][0] = 1
            label_tensor[i][S_i][S_j][1] = xmins_n[i][j]
            label_tensor[i][S_i][S_j][2] = ymins_n[i][j]
            label_tensor[i][S_i][S_j][3] = xmaxs_n[i][j]-xmins_n[i][j]
            label_tensor[i][S_i][S_j][4] = ymaxs_n[i][j]-ymins_n[i][j]
            label_tensor[i][S_i][S_j][self.B*5+class_num[i][j]] = 1
        return label_tensor

    def __load_image__(self,path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img,self.image_resize)
        img = img/127.5-1   #image normalization
        return img

    def train_data(self,batch_size):
        self.__getTrainPath__()
        train_pre_info = [self.__toLabels__(path) for path in self.train_labels_path]
        train_numered_info = self.__kindToNumber__(train_pre_info)
        train_label = self.__labelsNormalization__(train_numered_info[:5])
        train_label_datasets = tf.data.Dataset.from_tensor_slices(train_label)
        train_image_datasets = tf.data.Dataset.from_tensor_slices(self.train_images_path)
        train_image_datasets = train_image_datasets.map(self.__load_image__)
        train_dataset = tf.data.Dataset.zip((train_image_datasets,train_label_datasets))
        train_dataset = train_dataset.repeat().shuffle(len(self.train_images_name)).batch(batch_size)
        return train_dataset

    def test_data(self,batch_size):
        self.__getTestPath__()
        test_pre_info = [self.__toLabels__(path) for path in self.test_labels_path]
        test_numbered_info = self.__kindToNumber__(test_pre_info)
        test_label = self.__labelsNormalization__(test_numbered_info[:5])
        test_label_datasets = tf.data.Dataset.from_tensor_slices(test_label)
        test_image_datdasets = tf.data.Dataset.from_tensor_slices(self.test_images_path)
        test_image_datdasets = test_image_datdasets.map(self.__load_image__)
        test_dataset = tf.data.Dataset.zip((test_image_datdasets,test_label_datasets))
        test_dataset = test_dataset.repeat().batch(batch_size)
        return test_dataset

    def test(self,dataset):
        for img,label in dataset.take(2):
            plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
            x, y, w, h = [], [], [], []
            for i in range(label.shape[1]):
                for k in range(label.shape[2]):
                    if label[0][i][k][0] == 1:
                        x.append(label[0][i][k][1])
                        y.append(label[0][i][k][2])
                        w.append(label[0][i][k][3])
                        h.append(label[0][i][k][4])
                    else:
                        continue
            x = np.array()
            y = np.array(y)
            w = np.array(w)
            h = np.array(h)
            x, y, w, h = x * 224, y * 224, w * 224, h * 224
            dirc = {}
            for i in range(len(x)):
                dirc[str(i)] = Rectangle((x[i], y[i]), w[i], h[i], fill=False, color='red')
            ax = plt.gca()
            for i in range(len(x)):
                ax.axes.add_patch(dirc[str(i)])
            plt.show()

PATH = 'C:\\Users\\10372\\Desktop\\VOC\\VOCdevkit\\VOC2012'
DATA = LoadVOCdata(PATH,S = 14,B = 2,C = 20,image_resize=[224,224])
c = DATA.train_data(32)
DATA.test(c)

import cv2 as cv
from lxml import etree
import numpy as np
import os

path = 'C:\\Users\\10372\\Desktop\\VOC\\VOCdevkit\\VOC2012'

def to_labels(path):
    xml = open('{}'.format(path)).read()
    sel = etree.HTML(xml)
    xmins = np.array(sel.xpath('//object/bndbox/xmin/text()')).astype(np.int)
    ymins = np.array(sel.xpath('//object/bndbox/ymin/text()')).astype(np.int)
    xmaxs = np.array(sel.xpath('//object/bndbox/xmax/text()')).astype(np.int)
    ymaxs = np.array(sel.xpath('//object/bndbox/ymax/text()')).astype(np.int)
    classes = sel.xpath('//object/name/text()')
    return [xmins,ymins,xmaxs,ymaxs,classes]

def get_kinds(labels):
    output1,output2,output3,output4,output5 = list(zip(*labels))
    kind = []
    for i in range(len(output5)):
        for j in range(len(output5[i])):
            if output5[i][j] not in kind:
                kind.append(output5[i][j])
    return kind

def mkdir(names,path):
    for name in names:
        os.mkdir(path+'\\'+name)

def make_train_data(path):
    train_images_path = []
    train_labels_path = []
    train_names = open(path + "\\ImageSets\\Main\\train.txt", 'r')
    for name in train_names:
        train_images_path.append(path + "\\JPEGImages\\" + name.split('\n')[0] + ".jpg")
        train_labels_path.append(path + "\\Annotations\\" + name.split('\n')[0] + ".xml")
    train_labels = [to_labels(path) for path in train_labels_path]
    train_kinds = get_kinds(train_labels)
    train_kinds_path = 'C:\\Users\\10372\\Desktop\\image\\train'
    kind_number = {}
    for kind in train_kinds:
        kind_number[kind] = 0
    for i in range(len(train_images_path)):
        img = cv.imread(train_images_path[i])
        xmins, ymins, xmaxs, ymaxs ,classes= train_labels[i]
        for j in range(len(xmins)):
            cuted_img = img[ymins[j]:ymaxs[j], xmins[j]:xmaxs[j]]
            cuted_img = cv.resize(cuted_img,(224,224))
            image_path = train_kinds_path+'\\'+classes[j]+'\\'+str(kind_number[classes[j]])+'.jpg'
            print(image_path)
            cv.imwrite(image_path, cuted_img)
            kind_number[classes[j]] += 1

def make_test_data(path):
    test_images_path = []
    test_labels_path = []
    test_names = open(path + "\\ImageSets\\Main\\val.txt", 'r')
    for name in test_names:
        test_images_path.append(path + "\\JPEGImages\\" + name.split('\n')[0] + ".jpg")
        test_labels_path.append(path + "\\Annotations\\" + name.split('\n')[0] + ".xml")
    test_labels = [to_labels(path) for path in test_labels_path]
    test_kinds = get_kinds(test_labels)
    test_kinds_path = 'C:\\Users\\10372\\Desktop\\image\\test'
    kind_number = {}
    for kind in test_kinds:
        kind_number[kind] = 0
    for i in range(len(test_images_path)):
        img = cv.imread(test_images_path[i])
        xmins, ymins, xmaxs, ymaxs, classes = test_labels[i]
        for j in range(len(xmins)):
            cuted_img = img[ymins[j]:ymaxs[j], xmins[j]:xmaxs[j]]
            cuted_img = cv.resize(cuted_img, (224, 224))
            image_path = test_kinds_path + '\\' + classes[j] + '\\' + str(kind_number[classes[j]]) + '.jpg'
            print(image_path)
            cv.imwrite(image_path, cuted_img)
            kind_number[classes[j]] += 1

make_test_data(path)
# img = cv.imread('C:\\Users\\10372\\Desktop\\VOC\\VOCdevkit\\VOC2012\\JPEGImages\\2008_000008.jpg')
# print(img.shape)
# path = 'C:\\Users\\10372\\Desktop\\VOC\\VOCdevkit\\VOC2012\\Annotations\\2008_000008.xml'
# xml = open('{}'.format(path)).read()
# sel = etree.HTML(xml)
# height = int(sel.xpath('//size/height/text()')[0])
# width = int(sel.xpath('//size/width/text()')[0])
# xmins = np.array(sel.xpath('//object/bndbox/xmin/text()')).astype(np.int)
# ymins = np.array(sel.xpath('//object/bndbox/ymin/text()')).astype(np.int)
# xmaxs = np.array(sel.xpath('//object/bndbox/xmax/text()')).astype(np.int)
# ymaxs = np.array(sel.xpath('//object/bndbox/ymax/text()')).astype(np.int)
# index = 0
# print(xmins)
# cuted_img = img[ymins[index]:ymaxs[index],xmins[index]:xmaxs[index]]
# cv.imwrite('hourse\hourse_1.jpg',cuted_img)
# cv.imshow('image',cuted_img)
# cv.imshow('image',img)
# cv.waitKey(0)
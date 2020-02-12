import cv2
import os
import xml.etree.ElementTree as pars
import imutils
import numpy as np


def load_images_from_folder(folder):
    im_list = []

    for filename in os.listdir(folder + "/images/"):
        img = cv2.imread(os.path.join(folder + "/images/" + filename))
        full_name = filename.split('.')
        part_name = full_name[0]

        if not img is None:
            im_list.append((part_name, img))

    return im_list


def load_objects(im_list, folder):
    object_list = []

    for pic in im_list:
        part_name = pic[0]
        img = pic[1]

        e = pars.parse(folder + '/annotations/xmls/' + part_name + '.xml')

        root = e.getroot()

        objects = []
        for object in root.findall('object'):
            name = object.find('name').text
            for box in object.findall('bndbox'):
                points = [int(box.find('xmin').text),
                          int(box.find('ymin').text),
                          int(box.find('xmax').text),
                          int(box.find('ymax').text)]
            objects.append(points)
        object_list.append([img, part_name, objects])

    # print(object_list[0][2][0][1])
    return object_list
    # return object_list


def detector(rgb_image):
    """Детектирование пешеходов на изображении:
    Для каждого изображения формируется массив, состоящий из координат углов объектов(xmin,ymin, xmax, ymax) - координаты левого верхнего угла и
    правого нижнего угла объекта
    rects = [[335, 184, 384, 267]]
    """
    rects = [[100, 100, 200, 250], [140, 120, 200, 250]]
    return rects


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects = []
new = load_objects(load_images_from_folder("/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val"),
"/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val")
# print(h.load_images_from_folder("data/val"))
# print("--------")
counter = 0
nbins = 9
cellSize = (8, 8)
blockSize = (16, 16)
blockStride = (8, 8)
winSize = (64, 128)
winStride = (10, 10)
padding = (16, 16)
scale = 1.045
meanShift = -1
meanShift = True if meanShift > 0 else False

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

for img in new:
    img[0] = cv2.resize(img[0], (img[0].shape[1], img[0].shape[0]))

    (a, weights) = hog.detectMultiScale(img[0], winStride=winStride, padding=padding,scale=scale, useMeanshiftGrouping=meanShift)
    rects.append()
print(rects)
    #for (x, y, w, h) in rects:
    #    cv2.rectangle(img[0], (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imshow(str(counter), img[0])
    #counter += 1


while True:
    key = cv2.waitKey()
    if key == ord("f"):
        break
cv2.destroyAllWindows()

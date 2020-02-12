import cv2
import imutils
import numpy as np
import helpers

# модуль для детектирования

# если вы используте модель для детектирования добавьте ее здесь:
#################################################################


#################################################################
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

new = helpers.load_objects(helpers.load_images_from_folder("/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val"),
                           "/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val")
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


def detector(img):
    """Детектирование пешеходов на изображении:
    Для каждого изображения формируется массив, состоящий из координат углов объектов(xmin,ymin, xmax, ymax) - координаты левого верхнего угла и
    правого нижнего угла объекта
    rects = [[335, 184, 384, 267]]
    """

    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    (rects, weights) = hog.detectMultiScale(img, winStride=winStride, padding=padding, scale=scale,
                                            useMeanshiftGrouping=meanShift)

    for i in range(len(rects)):
        rects[i][2] += rects[i][0]
        rects[i][3] += rects[i][1]
    return rects


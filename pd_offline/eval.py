import cv2
import imutils
import helpers as h
import numpy as np

# модуль для детектирования

# если вы используте модель для детектирования добавьте ее здесь:
#################################################################


#################################################################


def detector(rgb_image):
    """Детектирование пешеходов на изображении:
    Для каждого изображения формируется массив, состоящий из координат углов объектов(xmin,ymin, xmax, ymax) - координаты левого верхнего угла и
    правого нижнего угла объекта
    rects = [[335, 184, 384, 267]]
    """
    rects = [[100, 100, 200, 250], [140, 120, 200, 250]]
    return rects


new = h.load_objects(h.load_images_from_folder("/Desktop/pd_offline/data/val"), "/Desktop/pd_offline/data/val")
# print(h.load_images_from_folder("data/val"))
# print("--------")
cv2.imshow("gg", new[0][0][0])
while True:
    key = cv2.waitKey()
    if key == ord("f"):
        break
cv2.destroyAllWindows()

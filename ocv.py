import cv2
import os
import xml.etree.ElementTree as pars
import imutils


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

new = load_objects(load_images_from_folder("/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val"),
                   "/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val")
# print(h.load_images_from_folder("data/val"))
# print("--------")
counter = 0
for img in new:
    img[0] = cv2.resize(img[0], (1152, 864))
    (rects, weights) = hog.detectMultiScale(img[0], scale=1.0656, winStride=(2, 2))
    for (x, y, w, h) in rects:
        cv2.rectangle(img[0], (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow(str(counter), img[0])
    counter += 1

# img1 = cv2.imread("/Users/nikita/PycharmProjects/mtmocv/pd_offline/data/val/images/AAFxkB2.jpg")
# gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,7),0)
# img1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
# (rects, weights) = hog.detectMultiScale(img1, scale=1.065, winStride=(1, 1))
# for (x, y, w, h) in rects:
#     cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
# cv2.imshow(str(1), img1)

while True:
    key = cv2.waitKey()
    if key == ord("f"):
        break
cv2.destroyAllWindows()

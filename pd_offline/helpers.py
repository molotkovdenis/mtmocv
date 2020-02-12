import cv2
import os
import xml.etree.ElementTree as pars

def load_images_from_folder(folder):

    im_list = []

    for filename in os.listdir(folder + "/images/"):
        img = cv2.imread(os.path.join(folder+"/images/" + filename))
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

import eval
import helpers
import cv2


def load_data():
    try:
        image_dir_test = "data/train/"
        test_image_list = helpers.load_images_from_folder(image_dir_test)
        test_object_list = helpers.load_objects(test_image_list, image_dir_test)
    except Exception as e:
        test_object_list = []

    return test_object_list


def predict(frame):
    predicted_label = eval.detector(frame)
    return predicted_label


def iou(points, predicted_points):
    """IoU = Area of overlap / Area of union
    Площадь пересечеения 2 знаков / общая площадь 2 знаков
    Значение выше 0.5 считаестя удачным детектированием знака
    """

    xA = max(points[0], predicted_points[0])
    yA = max(points[1], predicted_points[1])
    xB = min(points[2], predicted_points[2])
    yB = min(points[3], predicted_points[3])

    # x1, y1, xw1, yh1 = points
    # x2, y2, xw2, yh2 = predicted_points
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles

    boxAArea = (points[2] - points[0] + 1) * (points[3] - points[1] + 1)
    boxBArea = (predicted_points[2] - predicted_points[0] + 1) * (predicted_points[3] - predicted_points[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def check_prediction(test_image_label, predict_image_label):
    """ Проверка детектирования на изображении.
        Двойной проход по изображениям знака, обеспечивающий сравнение каждого существующего
        знака со всеми предсказанными по очереди и оперделяющий знак при наличии совпадения.
        Также при отсутствии совпадения определяющий что данный знак отсутствует и формирующй
        массив ложных совпадений
    """

    true_predicted = []
    false_predicted = []

    for p_obj in predict_image_label:

        predicted = []

        for obj in test_image_label:
            iou_res = iou(obj, p_obj)
            predicted.append(iou_res)

        pred = None
        for i in predicted:
            if i > 0.35:
                pred = True
                break
            else:
                pred = False

        if pred == True:
            true_predicted.append(1)
        else:
            false_predicted.append(1)



    if len(true_predicted) >= len(test_image_label):
        predicted_count = len(test_image_label)
    else:
        predicted_count = len(true_predicted)

    local_acc = predicted_count / len(test_image_label)

    false_positive = len(false_predicted) / len(test_image_label)

    return local_acc, false_positive


def get_misclassified_images(test_images):
    """ Проверка массива изображений на совпадение
    Результат: среднее точности и среднееточности срабатывания"""
    local_acc_summ = 0.
    false_positive_summ = 0.
    count = 0.

    for frame in test_images:
        # cv2.imshow("frame",frame)
        # cv2.waitKey(0)
        count += 1

        test_image_label = frame[2]
        predict_image_label = predict(frame[0])

        local_acc, local_acc_false_positive = check_prediction(test_image_label, predict_image_label)

        local_acc_summ += local_acc
        false_positive_summ += local_acc_false_positive

    return (local_acc_summ / count, false_positive_summ / count)


test_object_list = load_data()
# print(test_object_list)


MISCLASSIFIED = get_misclassified_images(test_object_list)
print("Точность: {}, Ложное предсказание (false positive): {}".format(MISCLASSIFIED[0], MISCLASSIFIED[1]))
fin_acc = MISCLASSIFIED[0]-MISCLASSIFIED[1]
if fin_acc < 0:
    fin_acc = 0
print("Финальная точность: {}".format(fin_acc))

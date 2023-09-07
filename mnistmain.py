# Recognition of handwritten numbers by picture

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

mnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

def mnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(mnist_labels[result[0]])


def numbers_extract(image_file: str, out_size=28) -> list[any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours / Получаем контуры
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    numbers = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            number_crop = gray[y:y + h, x:x + w]
            # print(number_crop.shape)

            # Resize number canvas to square / Изменение размера холста (числа) на квадрат
            size_max = max(w, h)
            number_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom / Увеличение изображения сверху вниз
                y_pos = size_max//2 - h//2
                number_square[y_pos:y_pos + h, 0:w] = number_crop
            elif w < h:
                # Enlarge image left-right / Увеличение изображения влево-вправо
                x_pos = size_max//2 - w//2
                number_square[0:h, x_pos:x_pos + w] = number_crop
            else:
                number_square = number_crop

            # Resize number to 28x28 and add number and its X-coordinate / Изменение размера числа на 28x28 и добавление числа и его координату X.
            numbers.append((x, w, cv2.resize(number_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate / Сортировка массива по координате X
    numbers.sort(key=lambda x: x[0], reverse=False)
    # Examination / Проверка
    # cv2.imshow('1', numbers[1][2])
    # cv2.imshow('2', numbers[2][2])
    # cv2.waitKey(0)
    return numbers


def img_numbers_to_str(model: any, image_file: str):
    # Function to convert image to string / Функция для преобразования картинки в строку
    numbers = numbers_extract(image_file)
    s_out = ""
    for i in range(len(numbers)):
        dn = numbers[i+1][0] - numbers[i][0] - numbers[i][1] if i < len(numbers) - 1 else 0
        s_out += mnist_predict_img(model, numbers[i][2])
        if (dn > numbers[i][1]/4):
            s_out += ' '
    return s_out

# model = keras.models.load_model('mnist_model.keras')
# s_out = img_to_str(model, "image.png")
# print(s_out)
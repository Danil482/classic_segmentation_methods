import cv2
import numpy as np


def global_binarization(input_path, output_path, threshold):
    # Загрузка изображения
    image = cv2.imread(input_path)
    if image is None:
        print("Ошибка: не удалось загрузить изображение по пути:", input_path)
        return

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Выполнение глобальной бинаризации без cv2.threshold:
    # Создаем бинарное изображение: если значение пикселя >= threshold, то 255, иначе 0.
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)

    # Сохранение бинаризованного изображения
    cv2.imwrite(output_path, binary)
    print(f"Бинаризованное изображение сохранено как {output_path}")


if __name__ == '__main__':
    global_binarization(
        input_path='C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg',
        output_path='C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/2.jpg',
        threshold=155
    )



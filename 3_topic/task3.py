import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_sobel(gray):
    """
    Применяет оператор Собеля по горизонтали и вертикали к изображению и объединяет результаты.
    """
    # Оператор Собеля по горизонтали (dx=1, dy=0)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Оператор Собеля по вертикали (dx=0, dy=1)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисляем абсолютные значения и преобразуем к типу uint8
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Объединяем результаты, используя взвешенное сложение
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return sobel_combined


def apply_canny(gray, lower_threshold=50, upper_threshold=150):
    """
    Применяет оператор Canny для обнаружения границ к изображению.
    """
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    return edges


def process_image(image_path, lower_canny=50, upper_canny=150):
    """
    Загружает изображение, переводит его в оттенки серого и применяет операторы Собеля и Canny.
    Возвращает исходное изображение, результат Собеля и результат Canny.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")
    # Преобразование в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Применяем оператор Собеля
    sobel_result = apply_sobel(gray)
    # Применяем оператор Canny
    canny_result = apply_canny(gray, lower_threshold=lower_canny, upper_threshold=upper_canny)

    return img, sobel_result, canny_result


# Пусть у нас есть два изображения:
landscape_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/0001.jpg'  # изображение пейзажа
objects_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg'  # изображение с объектами

# Обработка изображений
landscape_img, landscape_sobel, landscape_canny = process_image(landscape_path, 50, 150)
objects_img, objects_sobel, objects_canny = process_image(objects_path, 50, 150)

# Визуализация результатов
plt.figure(figsize=(18, 12))

# Пейзаж
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(landscape_img, cv2.COLOR_BGR2RGB))
plt.title("Пейзаж: Исходное изображение")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(landscape_sobel, cmap='gray')
plt.title("Пейзаж: Собель")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(landscape_canny, cmap='gray')
plt.title("Пейзаж: Canny")
plt.axis('off')

# Изображение с объектами
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(objects_img, cv2.COLOR_BGR2RGB))
plt.title("Объекты: Исходное изображение")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(objects_sobel, cmap='gray')
plt.title("Объекты: Собель")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(objects_canny, cmap='gray')
plt.title("Объекты: Canny")
plt.axis('off')

plt.tight_layout()
plt.show()

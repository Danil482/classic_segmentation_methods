import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def region_growing(gray, seed_point, threshold):
    """
    Функция для сегментации изображения методом Region Growing.

    Аргументы:
        gray       -- изображение в оттенках серого (numpy.ndarray)
        seed_point -- начальная точка (tuple: (x, y))
        threshold  -- пороговое значение (int)

    Возвращает:
        region_mask -- бинарная маска (numpy.ndarray) с регионами,
                       принадлежащими области (значение 255) и остальными пикселями (0)
    """
    # Размеры изображения
    height, width = gray.shape

    # Создаем маску, где будем отмечать включенные в регион пиксели
    region_mask = np.zeros_like(gray, dtype=np.uint8)
    # Флаг посещения для каждого пикселя
    visited = np.zeros_like(gray, dtype=bool)

    # Инициализируем очередь для обхода (можно использовать deque)
    queue = deque()
    queue.append(seed_point)
    visited[seed_point[1], seed_point[0]] = True  # Обратите внимание: индексы [y, x]

    # Значение интенсивности начального пикселя (seed)
    seed_value = int(gray[seed_point[1], seed_point[0]])

    # Определяем соседей (8-связность)
    neighbors = [(-1, -1), (0, -1), (1, -1),
                 (-1, 0), (1, 0),
                 (-1, 1), (0, 1), (1, 1)]

    while queue:
        x, y = queue.popleft()
        region_mask[y, x] = 255  # Отмечаем пиксель как принадлежащий региону

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            # Проверяем границы изображения
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if visited[ny, nx]:
                continue
            # Если разница интенсивности не превышает порог, добавляем пиксель в регион
            if abs(int(gray[ny, nx]) - seed_value) <= threshold:
                queue.append((nx, ny))
            # Отмечаем пиксель как посещенный, даже если он не удовлетворяет условию
            visited[ny, nx] = True

    return region_mask


# Пример использования:
if __name__ == "__main__":
    # Загрузка изображения и преобразование его в оттенки серого
    img = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
    if img is None:
        raise FileNotFoundError("Изображение не найдено, проверьте путь.")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    seed_points = [(gray.shape[1] // 2, gray.shape[0] // 2), (0, 0), (150, 250), (20, 20), (70, 250)]
    for seed_point in seed_points:
        # Задаем порог (например, 20)
        threshold = 20

        # Выполнение Region Growing
        region_mask = region_growing(gray, seed_point, threshold)

        # Отображение результатов
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(img[..., ::-1])  # BGR -> RGB
        plt.title("Исходное изображение")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Изображение в оттенках серого")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(region_mask, cmap='gray')
        plt.title("Результат Region Growing")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

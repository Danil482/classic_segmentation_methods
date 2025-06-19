import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def region_growing(img, seeds, threshold=10):
    """
    Реализация наращивания регионов от seed-точек.

    :param img: входное серое изображение (2D numpy array)
    :param seeds: список координат точек [(x1,y1), (x2,y2), ...]
    :param threshold: максимальная разница интенсивностей для добавления пикселя в регион
    :return: метки регионов (2D массив того же размера, где каждому региону соответствует свой номер)
    """

    h, w = img.shape
    labels = np.zeros((h, w), dtype=np.int32)  # 0 - фон, >0 - регионы
    current_label = 1

    for seed in seeds:
        x_seed, y_seed = seed
        if labels[y_seed, x_seed] != 0:
            # Точка уже принадлежит какому-то региону
            continue

        region_pixels = [(x_seed, y_seed)]
        labels[y_seed, x_seed] = current_label
        mean_intensity = float(img[y_seed, x_seed])
        count = 1

        # Очередь для обхода соседей
        queue = [(x_seed, y_seed)]

        while queue:
            x, y = queue.pop(0)

            # Проверяем соседей 8-связность
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < w) and (0 <= ny < h):
                        if labels[ny, nx] == 0:
                            intensity = float(img[ny, nx])
                            # Проверяем условие включения в регион
                            if abs(intensity - mean_intensity) <= threshold:
                                labels[ny, nx] = current_label
                                queue.append((nx, ny))
                                # Обновляем среднее интенсивность региона
                                mean_intensity = (mean_intensity * count + intensity) / (count + 1)
                                count += 1

        current_label += 1

    return labels

if __name__ == "__main__":
    # Загрузка изображения и преобразование в оттенки серого
    image = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
    if image is None:
        raise FileNotFoundError("Изображение не найдено, проверьте путь.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Кол-во seed точек
    k = 55

    h, w = gray.shape

    # Генерируем k случайных seed точек (x,y)
    random.seed(42)  # для воспроизводимости
    seeds = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(k)]

    labels = region_growing(gray, seeds, threshold=15)

    # Создаем цветное изображение для визуализации результатов
    segmented_color = np.zeros((h, w, 3), dtype=np.uint8)

    # Генерируем цвета для каждого региона
    colors = []
    random.seed(42)
    for i in range(k + 1):
        colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    for y in range(h):
        for x in range(w):
            label = labels[y,x]
            if label > 0:
                segmented_color[y,x] = colors[label]
            else:
                segmented_color[y,x] = (0,0,0)  # черный фон

    # Отображаем исходное и сегментированное изображение
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title('Оригинальное изображение (оттенки серого)')
    plt.imshow(gray, cmap='gray')
    plt.scatter([s[0] for s in seeds], [s[1] for s in seeds], c='red', marker='x')  # показываем seed точки

    plt.subplot(1,2,2)
    plt.title('Сегментация методом Region Growing')
    plt.imshow(cv2.cvtColor(segmented_color, cv2.COLOR_BGR2RGB))

    plt.show()

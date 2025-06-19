import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения (замените путь на актуальный)
img = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
if img is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применение оператора Собеля по горизонтали (dx=1, dy=0)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
# Применение оператора Собеля по вертикали (dx=0, dy=1)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Вычисление абсолютных значений градиентов и преобразование их к типу uint8
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

# Объединение градиентов (например, путем суммирования)
sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# Отображение результатов с использованием matplotlib
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title("Оттенки серого")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(abs_sobel_x, cmap='gray')
plt.title("Собель по горизонтали")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(abs_sobel_y, cmap='gray')
plt.title("Собель по вертикали")
plt.axis('off')

plt.tight_layout()
plt.show()

# Дополнительно можно отобразить комбинированный результат
plt.figure(figsize=(6, 6))
plt.imshow(sobel_combined, cmap='gray')
plt.title("Объединенный результат (Собель)")
plt.axis('off')
plt.show()


# | Критерий              | Собель                     | Кенни                        |
# |-----------------------|----------------------------|------------------------------|
# | Сложность             | Простая                    | Сложная                      |
# | Скорость              | Быстрая                   | Медленнее                    |
# | Устойчивость к шуму   | Низкая                    | Высокая                     |
# | Качество краёв        | Широкие, менее точные      | Тонкие, чёткие, непрерывные |
# | Применение            | Быстрая предварительная обработка | Точная детекция границ       |
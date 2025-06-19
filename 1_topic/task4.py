import cv2
import numpy as np
import matplotlib.pyplot as plt

# Задаем путь к изображению и пороговое значение
input_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg'
threshold_value = 155

# Загрузка изображения
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Бинаризация с помощью встроенной функции cv2.threshold
# Здесь используется метод обычной бинаризации (без автоматического подбора порога)
ret, thresh_builtin = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
print(f"Бинаризация cv2.threshold выполнена с пороговым значением: {threshold_value}")

# 2. Бинаризация с использованием пользовательского алгоритма (без cv2.threshold)
# Если значение пикселя больше или равно threshold_value, то пиксель становится белым (255), иначе черным (0)
thresh_custom = np.where(gray >= threshold_value, 255, 0).astype(np.uint8)

# Отображение результатов с помощью matplotlib
plt.figure(figsize=(18, 6))

# Исходное изображение (цветное)
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

# Результат встроенной бинаризации cv2.threshold
plt.subplot(1, 3, 2)
plt.imshow(thresh_builtin, cmap='gray')
plt.title("Бинаризация (cv2.threshold)")
plt.axis('off')

# Результат пользовательской бинаризации
plt.subplot(1, 3, 3)
plt.imshow(thresh_custom, cmap='gray')
plt.title("Пользовательская бинаризация (np.where)")
plt.axis('off')

plt.show()

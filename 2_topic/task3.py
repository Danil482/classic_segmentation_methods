import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage

# Загрузка исходного изображения
image = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
if image is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Применяем адаптивную бинаризацию
# Параметры: block_size (размер окна, должен быть нечетным) и константа C
block_size = 13   # можно изменить для экспериментов
C = 2            # можно изменить для экспериментов
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, block_size, C)

# Применяем морфологические операции для удаления мелкого шума
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Выделяем уверенный фон путём дилатации
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Вычисляем distance transform – мера расстояния до ближайшего нулевого пикселя
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Применяем пороговую операцию к distance transform для выделения уверенного переднего плана.
# Здесь используется порог 0.7 от максимального значения – этот параметр можно экспериментально менять.
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Определяем неизвестную область: разница между фоном и передним планом
unknown = cv2.subtract(sure_bg, sure_fg)

# Маркировка уверенных областей переднего плана (seed markers)
ret, markers = cv2.connectedComponents(sure_fg)

# Увеличиваем метки, чтобы фон имел значение 1, а маркеры начинались с 2
markers = markers + 1

# Помечаем неизвестную область нулём
markers[unknown == 255] = 0

# Применяем алгоритм Watershed для сегментации
markers = cv2.watershed(image, markers)

# Отмечаем границы (где markers == -1) на исходном изображении (например, красным цветом)
segmented = image.copy()
segmented[markers == -1] = [255, 0, 0]

# Визуализация результатов
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title("Изображение в оттенках серого")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title(f"Адаптивная бинаризация\n(block_size={block_size}, C={C})")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sure_bg, cmap='gray')
plt.title("Уверенный фон")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(dist_transform, cmap='jet')
plt.title("Distance Transform")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Результат Watershed")
plt.axis('off')

plt.tight_layout()
plt.show()

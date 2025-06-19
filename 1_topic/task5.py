import cv2
import numpy as np
import matplotlib.pyplot as plt


def custom_adaptive_threshold(gray, block_size, C, method='gaussian'):
    """
    Пользовательская реализация адаптивной бинаризации.

    Аргументы:
    gray       -- изображение в градациях серого (numpy.ndarray)
    block_size -- размер окна (должен быть нечетным, например, 11)
    C          -- константа, вычитаемая из локального порога
    method     -- метод вычисления локального порога ('mean' или 'gaussian')

    Возвращает:
    бинаризованное изображение (numpy.ndarray, uint8)
    """
    # Приводим изображение к типу float32 для точности вычислений
    gray_float = gray.astype(np.float32)

    if method == 'mean':
        # Локальный порог – среднее значение в окне block_size x block_size
        local_thresh = cv2.blur(gray_float, (block_size, block_size))
    elif method == 'gaussian':
        # Локальный порог – взвешенное среднее (гауссовское размытие)
        local_thresh = cv2.GaussianBlur(gray_float, (block_size, block_size), 0)
    else:
        raise ValueError("Метод должен быть 'mean' или 'gaussian'")

    # Вычисляем итоговый бинарный результат: если значение пикселя >= (локальное_среднее - C) -> 255, иначе 0
    binary = np.where(gray_float >= (local_thresh - C), 255, 0).astype(np.uint8)
    return binary


# Загрузка изображения (замените путь на актуальный)
input_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg'
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Адаптивная бинаризация встроенной функцией OpenCV (используем метод Гауссова)
built_in = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# Пользовательская адаптивная бинаризация с разными значениями C
custom_C2 = custom_adaptive_threshold(gray, block_size=11, C=2, method='gaussian')
custom_C5 = custom_adaptive_threshold(gray, block_size=11, C=5, method='gaussian')
custom_C10 = custom_adaptive_threshold(gray, block_size=11, C=10, method='gaussian')

# Отображение исходного изображения и результатов бинаризации
plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(built_in, cmap='gray')
plt.title("cv2.adaptiveThreshold\n(гауссов, C=2)")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(custom_C2, cmap='gray')
plt.title("Custom adaptive (C=2)")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(custom_C5, cmap='gray')
plt.title("Custom adaptive (C=5)")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(custom_C10, cmap='gray')
plt.title("Custom adaptive (C=10)")
plt.axis('off')

plt.tight_layout()
plt.show()

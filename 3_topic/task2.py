import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения (замените путь на актуальный)
img = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
if img is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Определяем несколько пар пороговых значений для оператора Кэнни
thresholds = [
    (50, 150),
    (100, 200),
    (150, 250)
]

# Применяем оператор Кэнни для каждой пары порогов
edges_results = []
for lower, upper in thresholds:
    edges = cv2.Canny(gray, lower, upper)
    edges_results.append(edges)

# Отображение результатов
plt.figure(figsize=(18, 6))

# Исходное изображение
plt.subplot(1, len(thresholds)+1, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

# Результаты для каждого набора порогов
for idx, (edge_img, (lower, upper)) in enumerate(zip(edges_results, thresholds), start=2):
    plt.subplot(1, len(thresholds)+1, idx)
    plt.imshow(edge_img, cmap='gray')
    plt.title(f"Canny\n({lower}, {upper})")
    plt.axis('off')

plt.tight_layout()
plt.show()

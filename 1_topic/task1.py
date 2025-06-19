import cv2
import matplotlib.pyplot as plt

# Загрузка изображения (замените 'input.jpg' на путь к вашему файлу)
image = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg')
if image is None:
    raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Бинаризация с помощью глобальной пороговой сегментации (метод Отсу)
ret, thresh_global = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Определённый порог (Отсу): {ret}")

# Бинаризация с помощью адаптивного порога
# Здесь используется адаптивный метод Gaussian, блок (окно) 11x11 и постоянная C = 2
thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Отображение исходного и бинаризованных изображений
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(thresh_global, cmap='gray')
plt.title("Бинаризация (cv2.threshold, Отсу)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(thresh_adaptive, cmap='gray')
plt.title("Адаптивная бинаризация (cv2.adaptiveThreshold)")
plt.axis('off')

plt.show()

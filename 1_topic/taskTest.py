import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(img):
    """
    Выполняет три этапа предобработки:
    1. Деноизация с помощью fastNlMeansDenoisingColored.
    2. Гистограммная эквализация яркости (L-канала в LAB) для коррекции яркости и контраста.
    3. Цветовая нормализация с помощью cv2.normalize.

    Возвращает:
        denoised  - изображение после денойзинга,
        equalized - изображение после гистограммной эквализации,
        normalized- изображение после цветовой нормализации.
    """
    # 1. Деноизация
    img = cv2.resize(img, (640, 640), cv2.INTER_NEAREST)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    # 2. Гистограммная эквализация для коррекции яркости/контраста
    # Перевод изображения из BGR в LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Применяем гистограммную эквализацию к L-каналу
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    equalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return denoised, equalized


def preprocess_dataset(input_folder, output_folder):
    """
    Обрабатывает все изображения из input_folder, применяя функцию preprocess_image,
    и сохраняет обработанные изображения в output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ищем файлы с расширениями jpg, jpeg, png, bmp
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        _, _, processed_img = preprocess_image(img)
        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_folder, base_name)
        cv2.imwrite(save_path, processed_img)
        print(f"Processed and saved: {save_path}")


# Загрузка исходного изображения
img_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/0001.jpg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Изображение не найдено, проверьте путь.")

# Выполняем предобработку и получаем промежуточные результаты
denoised, equalized = preprocess_image(img)


# Функция для корректного отображения изображения в matplotlib (BGR -> RGB)
def display_image(title, image, cmap=None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap=cmap)
    plt.title(title)
    plt.axis('off')


# Выводим исходное изображение и результаты после каждого этапа обработки
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
display_image("Исходное изображение", img)

plt.subplot(1, 3, 2)
display_image("После денойзинга", denoised)

plt.subplot(1, 3, 3)
display_image("После гистограммной эквализации", equalized)

plt.tight_layout()
plt.show()

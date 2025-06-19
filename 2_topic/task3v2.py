import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.measure import label


def erode(img, k_s=3, it=1):
    kernel = np.ones((k_s, k_s), 'uint8')
    erode_img = cv2.erode(img, kernel, iterations=it)
    return erode_img


def dilate(img, k_s=3, it=1):
    kernel = np.ones((k_s, k_s), 'uint8')
    dilate_img = cv2.dilate(img, kernel, iterations=it)
    return dilate_img


def del_small_areas(thresh, area_black=100, area_white=100):
    result = morphology.remove_small_objects(label(thresh), area_white)
    result[result > 0] = 255
    result = morphology.remove_small_objects(label(255 - result), area_black)
    result[result > 0] = 255
    result = 255 - result
    return result


def watershed_for_channel(channel_img):
    # Адаптивный порог (инвертированный бинарный)
    thresh_local_1 = cv2.adaptiveThreshold(channel_img, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           5, 25)

    # Морфологическая обработка для удаления шумов и мелких объектов
    thresh_local = del_small_areas(erode(dilate(thresh_local_1)))

    # Расчет расстояния до ближайшего нулевого пикселя (distance transform)
    distance = ndi.distance_transform_edt(thresh_local)

    # Поиск локальных максимумов на distance map — маркеры для watershed
    local_max = peak_local_max(distance,
                               min_distance=10,
                               footprint=np.ones((3, 3)),
                               labels=thresh_local.astype(np.int32))

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_max.T)] = True
    markers, _ = ndi.label(mask)

    # Запуск watershed сегментации (отрицание distance для поиска впадин)
    labels = watershed(-distance, markers, mask=thresh_local)

    return labels


def combine_labels(labels_channels):
    # labels_channels - список из 3 массивов одинакового размера (R,G,B)
    h, w = labels_channels[0].shape
    combined = np.stack(labels_channels, axis=-1)  # shape (h,w,3)

    # Преобразуем каждый кортеж меток в одно число — с помощью view или structured array
    # Для простоты — сделаем строковое представление и потом факторизируем
    flat_labels = combined.reshape(-1, 3)
    tuples = [tuple(x) for x in flat_labels]

    # Факторизация уникальных комбинаций
    unique_tuples, inverse = np.unique(tuples, return_inverse=True, axis=0)

    # Итоговое изображение с метками
    result = inverse.reshape(h, w)

    return result


if __name__ == "__main__":
    img = cv2.imread('C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/leaf.jpg')
    if img is None:
        raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")

    # OpenCV читает в формате BGR — меняем порядок на RGB для удобства
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = cv2.split(img_rgb)  # R,G,B каналы

    labels_channels = []
    contours_images = []

    for i, ch in enumerate(channels):
        labels = watershed_for_channel(ch)
        labels_channels.append(labels)

        # Нарисуем контуры сегментов для каждого канала
        contour_img = cv2.cvtColor(ch.copy(), cv2.COLOR_GRAY2RGB)
        for label_val in np.unique(labels):
            if label_val == 0:
                continue
            mask = np.zeros(labels.shape, dtype="uint8")
            mask[labels == label_val] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 1)
        contours_images.append(contour_img)

    # Визуализация результатов
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    channel_names = ['Red', 'Green', 'Blue']

    for i in range(3):
        ax = axes[0,i]
        ax.imshow(channels[i], cmap='gray')
        ax.set_title(f'{channel_names[i]} channel')
        ax.axis('off')

        ax = axes[1,i]
        ax.imshow(labels_channels[i], cmap='nipy_spectral')
        ax.set_title(f'Watershed labels ({channel_names[i]})')
        ax.axis('off')

        ax = axes[2,i]
        ax.imshow(contours_images[i])
        ax.set_title(f'Contours ({channel_names[i]})')
        ax.axis('off')

        # Дополнительно: показать оригинальный канал с наложенными seed точками (маркерами)
        # Можно добавить при необходимости

    # Последняя строка — оригинальное RGB изображение для сравнения
    for i in range(3):
        axes[3,i].imshow(img_rgb)
        axes[3,i].set_title('Original RGB image')
        axes[3,i].axis('off')

    plt.tight_layout()
    plt.show()

    # Использование:
    final_labels = combine_labels(labels_channels)

    plt.figure(figsize=(8, 6))
    plt.imshow(final_labels, cmap='nipy_spectral')
    plt.title('Combined segmentation labels')
    plt.axis('off')
    plt.show()

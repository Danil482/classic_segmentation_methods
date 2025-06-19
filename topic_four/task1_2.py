import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_segmentation(image, n_clusters=3, max_iterations=300):
    pixel_values = image.reshape((-1, image.shape[2])) if image.ndim == 3 else image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iterations, n_init=10)
    kmeans.fit(pixel_values)

    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    segmented_img = centers[labels.flatten()].reshape(image.shape)

    return segmented_img


if __name__ == '__main__':
    # Путь к изображению и проверка
    img_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/abc.jpg'
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Приводим к RGB один раз

    # Параметры
    clusters_list = [2, 3, 4, 5]
    iterations_list = [10, 50, 100, 300]

    # Папка для результатов
    output_dir = "C:/Users/dania/PycharmProjects/pythonProject/segmentation/results"
    os.makedirs(output_dir, exist_ok=True)

    # Обработка и визуализация
    for clusters in clusters_list:
        for iters in iterations_list:
            segmented = kmeans_segmentation(img, n_clusters=clusters, max_iterations=iters)

            # Отображение результата
            plt.figure(figsize=(6, 6))
            plt.imshow(segmented)
            plt.title(f"KMeans Segmentation: {clusters} clusters, {iters} iterations")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            # Сохранение
            result_path = os.path.join(output_dir, f"seg_clusters{clusters}_iters{iters}.png")
            plt.imsave(result_path, segmented)
            print(f"Сегментированное изображение сохранено: {result_path}")

    # оптитимум 4 кластера 100 итераций
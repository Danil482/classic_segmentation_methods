import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from skimage import data
import matplotlib.pyplot as plt


def dbscan_segmentation(img, eps=0.3, min_samples=3, return_clusters=False):
    # Уменьшаем изображение в 2 раза
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Проверяем, является ли изображение Grayscale или цветным
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        img = img.reshape(h, w, 1)  # Преобразуем Grayscale в трёхмерный массив с 1 каналом

    # Квантование цветов: округляем до ближайшего кратного 32
    img_quant = (img // 32) * 32

    # Формируем массив X в зависимости от количества каналов
    if img.shape[2] == 1:
        X = img_quant.reshape(-1, 1)  # Для Grayscale
    else:
        X = img_quant.reshape(-1, 3)  # Для цветных изображений

    X_scaled = StandardScaler().fit_transform(X)

    # Сэмплирование: 20% пикселей
    num_pixels = X_scaled.shape[0]
    sample_ratio = 0.2
    sample_idx = np.random.choice(num_pixels, int(num_pixels * sample_ratio), replace=False)
    X_sampled = X_scaled[sample_idx]

    # Кластеризация на подвыборке
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_sampled)
    sample_labels = db.labels_

    # Назначаем метки по ближайшим цветам
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_sampled, sample_labels)
    full_labels = knn.predict(X_scaled)

    n_clusters = len(set(full_labels)) - (1 if -1 in full_labels else 0)

    # Формируем результирующее изображение
    if img.shape[2] == 1:
        clustered_img = full_labels.reshape(h, w) * 32  # Масштабируем обратно для Grayscale
    else:
        clustered_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                label = full_labels[i * w + j]
                clustered_img[i, j] = img_quant[i, j] if label != -1 else [0, 0, 0]

    if return_clusters:
        return clustered_img, n_clusters
    else:
        return clustered_img


if __name__ == '__main__':
    # Загрузка изображения (RGB)
    img = data.coffee()

    # Список параметров
    eps_values = [0.3, 0.5, 0.7]
    min_samples_values = [3, 5]

    # Построение кластеризаций
    fig, axs = plt.subplots(len(eps_values), len(min_samples_values), figsize=(12, 8))
    fig.suptitle("Быстрая DBSCAN по цвету (с сэмплированием и квантованием)", fontsize=16)

    for i, eps in enumerate(eps_values):
        for j, min_samples in enumerate(min_samples_values):
            clustered_img, n_clusters = dbscan_segmentation(
                img=img,
                eps=eps,
                min_samples=min_samples,
                return_clusters=True
            )

            axs[i, j].imshow(clustered_img)
            axs[i, j].set_title(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('C:/Users/dania/PycharmProjects/pythonProject/segmentation/results/DBSCAN_optimization.png')
    plt.show()

import cv2
import numpy as np
from skimage import data
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from segmentation.topic_four.task1_2 import kmeans_segmentation
from segmentation.topic_four.task4 import dbscan_segmentation


def mean_shift_segmentation(image):
    # Формируем массив pixels в зависимости от количества измерений
    if len(image.shape) == 2:  # Grayscale
        pixels = image.reshape(-1, 1)
    else:  # Цветное изображение
        pixels = image.reshape(-1, 3)
    bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=50)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels)
    labels_ms = ms.labels_
    centers_ms = ms.cluster_centers_.astype(np.uint8)
    segmented_image_ms = centers_ms[labels_ms].reshape(image.shape)
    return segmented_image_ms


def segment_and_display(image, image_name):
    k = 3
    segmented_image_kmeans = kmeans_segmentation(image, n_clusters=k)

    segmented_image_ms = mean_shift_segmentation(image=image)

    segmented_image_db, n_clusters = dbscan_segmentation(img=image, return_clusters=True)
    print(f'Номер кластеров{n_clusters}')
    # Display the results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title(f'Исходное ({image_name})')
    axes[1].imshow(segmented_image_kmeans)
    axes[1].set_title('K-Means')
    axes[2].imshow(segmented_image_ms)
    axes[2].set_title('Mean Shift')
    axes[3].imshow(segmented_image_db)
    axes[3].set_title('DBSCAN')
    for ax in axes:
        ax.axis('off')
    plt.savefig(f'C:/Users/dania/PycharmProjects/pythonProject/segmentation/results/{image_name}_segmentation.png')
    plt.show()


if __name__ == "__main__":
    # Load sample images from scikit-image
    images = [
        (data.coffee(), 'Coffee'),
        (data.astronaut(), 'Astronaut'),
        (data.chelsea(), 'Chelsea')
    ]
    img_path = 'C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/langs.jpg'
    langs = cv2.imread(img_path)
    images = [
        (langs, 'Langs')
    ]

    # Perform segmentation and display for each image
    for img, name in images:
        segment_and_display(img, name)

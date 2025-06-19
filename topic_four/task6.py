import cv2
import numpy as np
from skimage import data
from segmentation.topic_four.task1_2 import kmeans_segmentation
from segmentation.topic_four.task4 import dbscan_segmentation
from segmentation.topic_four.task5 import mean_shift_segmentation
import matplotlib.pyplot as plt


def convert_to_color_space(image, color_space):
    if color_space == 'RGB':
        return image
    elif color_space == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif color_space == 'CIELab':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif color_space == 'CIELuv':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    else:
        raise ValueError("Unknown color space")


def plot_segmented_image(ax, segmented_image, color_space):
    if color_space == 'Grayscale':
        if segmented_image.dtype == np.float32 or segmented_image.dtype == np.float64:
            ax.imshow(segmented_image / 255.0, cmap='gray')
        else:
            ax.imshow(segmented_image, cmap='gray')
    elif color_space == 'RGB':
        if segmented_image.dtype == np.float32 or segmented_image.dtype == np.float64:
            ax.imshow(segmented_image / 255.0)
        else:
            ax.imshow(segmented_image)
    elif color_space == 'CIELab':
        segmented_rgb = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
        ax.imshow(segmented_rgb)
    elif color_space == 'CIELuv':
        segmented_rgb = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_LUV2RGB)
        ax.imshow(segmented_rgb)


color_spaces = ['Grayscale', 'RGB', 'CIELab', 'CIELuv']
methods = ['DBSCAN', 'K-Means', 'Mean Shift']
images = [
    (data.coffee(), 'coffee'),
    (data.astronaut(), 'astronaut'),
    (data.chelsea(), 'chelsea')
]

for img, img_name in images:
    image = getattr(data, img_name)()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Сегментация изображения {img_name}', fontsize=16)
    for i, method in enumerate(methods):
        for j, color_space in enumerate(color_spaces):
            ax = axes[i, j]
            converted_image = convert_to_color_space(image, color_space)
            if method == 'K-Means':
                segmented_image = kmeans_segmentation(converted_image)
            elif method == 'Mean Shift':
                segmented_image = mean_shift_segmentation(converted_image)
            elif method == 'DBSCAN':
                segmented_image = dbscan_segmentation(converted_image)
            plot_segmented_image(ax, segmented_image, color_space)
            ax.set_title(f'{method} в {color_space}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'C:/Users/dania/PycharmProjects/pythonProject/segmentation/results/{img_name}_segmentation_colors.png')
    plt.show()

# Анализ сегментации в четырёх цветовых пространствах и трёх методах показывает,
# что CIELab и CIELuv обеспечивают более точное разделение ярких цветов благодаря перцептивной равномерности,
# в то время как Grayscale теряет цветовую информацию, а RGB менее эффективен для сложных цветовых различий.
# Среди методов K-Means прост и быстр, Mean Shift адаптивен, а DBSCAN эффективен для шумных данных.
# Выбор комбинации зависит от конкретной задачи и характеристик изображения.
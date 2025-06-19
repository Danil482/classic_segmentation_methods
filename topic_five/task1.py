from skimage import data, segmentation, color
from skimage import graph
import matplotlib.pyplot as plt


def segment_and_display(img, compactness, n_segments):
    labels1 = segmentation.slic(img, compactness=compactness, n_segments=n_segments)
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title('Оригинал')
    ax[1].imshow(out1)
    ax[1].set_title(f'SLIC: c={compactness}, n={n_segments}')
    ax[2].imshow(out2)
    ax[2].set_title('Normalized Cut')
    for a in ax:
        a.axis('off')
    plt.show()


if __name__ == '__main__':
    img = data.coffee()
    # Исходные параметры
    segment_and_display(img, 15, 50)
    # Меньшая компактность
    segment_and_display(img, 5, 50)
    # Большая компактность
    segment_and_display(img, 30, 50)
    # Меньше сегментов
    segment_and_display(img, 15, 20)
    # Больше сегментов
    segment_and_display(img, 15, 100)

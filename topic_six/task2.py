import skimage.segmentation as seg
from skimage import data, color
from matplotlib import pyplot as plt
from segmentation.topic_six.task1 import circle_points


if __name__ == '__main__':
    image = data.coffee()
    image_gray = color.rgb2gray(image)
    center = [120, 270]
    points = circle_points(resolution=300, center=center, radius=110)[:-1]
    alphas = [0.1, 0.2, 0.3]
    betas = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(len(alphas), len(betas), figsize=(12, 12))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            snake = seg.active_contour(image_gray, points, alpha=alpha, beta=beta)
            ax = axes[i, j]
            ax.imshow(image)
            ax.plot(points[:, 0], points[:, 1], '--r', lw=2)
            ax.plot(snake[:, 0], snake[:, 1], '-g', lw=2)
            ax.set_title(f'alpha={alpha}, beta={beta}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'C:/Users/dania/PycharmProjects/pythonProject/segmentation/results/segmentation_coffee123.png')
    plt.show()

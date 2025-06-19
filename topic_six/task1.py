import numpy as np
import skimage.segmentation as seg
from skimage import data, color
from matplotlib import pyplot as plt


def circle_points(resolution, center, radius):
    radians = np.linspace(0, 2 * np.pi, resolution)
    c = center[1] + radius * np.cos(radians)
    r = center[0] + radius * np.sin(radians)
    return np.array([c, r]).T


def rectangle_points(resolution, center, width, height):
    row_center, col_center = center
    row_min = row_center - height / 2
    row_max = row_center + height / 2
    col_min = col_center - width / 2
    col_max = col_center + width / 2
    points_per_side = resolution // 4
    cols_top = np.linspace(col_min, col_max, points_per_side)
    rows_top = np.full(points_per_side, row_min)
    cols_right = np.full(points_per_side, col_max)
    rows_right = np.linspace(row_min, row_max, points_per_side)
    cols_bottom = np.linspace(col_max, col_min, points_per_side)
    rows_bottom = np.full(points_per_side, row_max)
    cols_left = np.full(points_per_side, col_min)
    rows_left = np.linspace(row_max, row_min, points_per_side)
    cols = np.concatenate([cols_top, cols_right, cols_bottom, cols_left])
    rows = np.concatenate([rows_top, rows_right, rows_bottom, rows_left])
    return np.vstack([cols, rows]).T


def triangle_points(resolution, center, size):
    row_center, col_center = center
    theta = np.radians([0, 120, 240])
    p_cols = col_center + size * np.cos(theta)
    p_rows = row_center + size * np.sin(theta)
    points_per_side = resolution // 3
    side1 = np.linspace([p_cols[0], p_rows[0]], [p_cols[1], p_rows[1]], points_per_side)
    side2 = np.linspace([p_cols[1], p_rows[1]], [p_cols[2], p_rows[2]], points_per_side)
    side3 = np.linspace([p_cols[2], p_rows[2]], [p_cols[0], p_rows[0]], points_per_side)
    return np.vstack([side1, side2, side3])


def ellipse_points(resolution, center, a, b):
    row_center, col_center = center
    theta = np.linspace(0, 2 * np.pi, resolution)
    cols = col_center + a * np.cos(theta)
    rows = row_center + b * np.sin(theta)
    return np.vstack([cols, rows]).T


if __name__ == '__main__':
    image = data.coffee()
    image_gray = color.rgb2gray(image)
    center = [120, 270]
    shapes = [
        {'name': 'circle', 'func': circle_points, 'params': {'resolution': 300, 'center': center, 'radius': 110}},
        {'name': 'rectangle', 'func': rectangle_points,
         'params': {'resolution': 300, 'center': center, 'width': 220, 'height': 220}},
        {'name': 'triangle', 'func': triangle_points, 'params': {'resolution': 300, 'center': center, 'size': 110}},
        {'name': 'ellipse', 'func': ellipse_points, 'params': {'resolution': 300, 'center': center, 'a': 110, 'b': 55}},
    ]
    for shape in shapes:
        points = shape['func'](**shape['params'])[:-1]
        snake = seg.active_contour(image_gray, points, alpha=0.2, beta=0.5)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(image)
        ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-g', lw=3)
        ax.set_title(shape['name'])
        ax.axis('off')
        plt.show()

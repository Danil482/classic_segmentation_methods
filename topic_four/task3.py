import numpy as np
import matplotlib.pyplot as plt
import cv2

# Создание цветовой плоскости: фиксируем L* и варьируем a* и b*
L_fixed = 75  # светлота
a_range = np.linspace(-128, 127, 400)
b_range = np.linspace(-128, 127, 400)
a_grid, b_grid = np.meshgrid(a_range, b_range)

# Собираем Lab-изображение (H, W, 3)
lab_image = np.zeros((400, 400, 3), dtype=np.float32)
lab_image[:, :, 0] = L_fixed
lab_image[:, :, 1] = a_grid
lab_image[:, :, 2] = b_grid

# Переводим в RGB
lab_image_uint8 = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)

# Ограничим значения к диапазону [0, 1] для отображения
lab_image_clipped = np.clip(lab_image_uint8, 0, 1)

# Отображение
plt.figure(figsize=(8, 8))
plt.imshow(lab_image_clipped)
plt.title("Цветовая плоскость CIELab (a*, b*) при L* = 75")
plt.xlabel("a* (зелёный ↔ красный)")
plt.ylabel("b* (синий ↔ жёлтый)")
plt.grid(False)
plt.axis('on')
plt.show()

# Пространство CIELab стремится быть перцептивно равномерным,
# т.е. одинаковые расстояния между цветами соответствуют одинаковой разнице, воспринимаемой глазом.
#
# RGB неравномерно — разные участки цветового диапазона в RGB могут восприниматься человеком с разной чувствительностью.
#
# Визуализируя мы видим, как меняются цвета при изменении оттенков и насыщенности, сохраняя одну и ту же яркость.

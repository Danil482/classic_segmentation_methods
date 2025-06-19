import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class BinarizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Приложение бинаризации изображений")

        self.image = None  # исходное изображение (BGR)
        self.processed_image = None  # бинаризованное изображение

        # Кнопка для выбора изображения
        self.btn_select = tk.Button(master, text="Выбрать изображение", command=self.select_image)
        self.btn_select.pack(pady=5)

        # Фрейм для выбора метода бинаризации (радиокнопки)
        method_frame = tk.Frame(master)
        method_frame.pack(pady=5)
        self.method_var = tk.StringVar(value="global")
        tk.Radiobutton(method_frame, text="Глобальная бинаризация", variable=self.method_var, value="global").pack(
            side=tk.LEFT, padx=5)
        tk.Radiobutton(method_frame, text="Адаптивная бинаризация", variable=self.method_var, value="adaptive").pack(
            side=tk.LEFT, padx=5)

        # Параметры для глобальной бинаризации: пороговое значение
        self.thresh_label = tk.Label(master, text="Порог (0-255):")
        self.thresh_label.pack()
        self.thresh_entry = tk.Entry(master, width=10)
        self.thresh_entry.insert(0, "127")
        self.thresh_entry.pack(pady=3)

        # Параметры для адаптивной бинаризации: размер блока и константа C
        self.block_label = tk.Label(master, text="Размер блока (нечетное число):")
        self.block_label.pack()
        self.block_entry = tk.Entry(master, width=10)
        self.block_entry.insert(0, "11")
        self.block_entry.pack(pady=3)

        self.C_label = tk.Label(master, text="Константа C:")
        self.C_label.pack()
        self.C_entry = tk.Entry(master, width=10)
        self.C_entry.insert(0, "2")
        self.C_entry.pack(pady=3)

        # Кнопка для выполнения бинаризации
        self.btn_process = tk.Button(master, text="Бинаризовать", command=self.binarize_image)
        self.btn_process.pack(pady=10)

        # Метка для отображения изображения
        self.image_label = tk.Label(master)
        self.image_label.pack(pady=10)

    def select_image(self):
        # Открываем диалог выбора файла
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                               filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")])
        if file_path:
            # Загружаем изображение с помощью OpenCV (BGR)
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.show_image(self.image)

    def show_image(self, cv_img):
        # Преобразуем изображение из BGR в RGB и создаём объект PIL.Image для отображения в Tkinter
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img  # сохраняем ссылку на изображение

    def binarize_image(self):
        if self.image is None:
            return

        # Преобразуем исходное изображение в оттенки серого
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        method = self.method_var.get()

        if method == "global":
            # Получаем пороговое значение из поля ввода
            try:
                thresh_val = int(self.thresh_entry.get())
            except ValueError:
                thresh_val = 127
            # Выполняем глобальную бинаризацию вручную (без cv2.threshold)
            binary = np.where(gray >= thresh_val, 255, 0).astype(np.uint8)
        elif method == "adaptive":
            # Получаем параметры для адаптивной бинаризации
            try:
                block_size = int(self.block_entry.get())
                # Размер блока должен быть нечетным, иначе прибавляем 1
                if block_size % 2 == 0:
                    block_size += 1
            except ValueError:
                block_size = 11
            try:
                C = int(self.C_entry.get())
            except ValueError:
                C = 2
            # Выполняем адаптивную бинаризацию с использованием cv2.adaptiveThreshold
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C)
        else:
            # Если метод не выбран, просто возвращаем исходное изображение
            binary = gray

        # Для отображения преобразуем бинарное изображение в трехканальное (BGR)
        display_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.processed_image = binary
        self.show_image(display_img)

        # Сохраняем результат в файл (например, "binarized_output.jpg")
        output_path = "C:/Users/dania/PycharmProjects/pythonProject/segmentation/images/binarized_output.jpg"
        cv2.imwrite(output_path, binary)
        print(f"Бинаризованное изображение сохранено как {output_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = BinarizationApp(root)
    root.mainloop()

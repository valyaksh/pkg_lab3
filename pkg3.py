# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# image_path = "/Users/sonyands/Desktop/bsu/лаб пкг/pkg3/6273737733.jpg"


# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Операция: Добавление целочисленной константы к изображению
# constant_value = 50
# added_image = cv2.add(image, constant_value)

# # Операция: Преобразование изображения в негатив
# negative_image = 255 - image

# # Операция: Умножение изображения на константу
# constant_multiplier = 2.0
# multiplied_image = cv2.multiply(image, constant_multiplier)

# # Операция: Степенное преобразование
# power = 0.5  
# power_transformed_image = np.power(image, power)

# # Операция: Логарифмическое преобразование
# log_transformed_image = np.log1p(image)

# # Построение гистограммы изображения
# hist, bins = np.histogram(image, bins=256, range=(0, 256))

# # Эквализация гистограммы
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max() / cdf.max()

# # Линейное контрастирование
# alpha = 1.5  
# beta = 20   
# contrast_image = alpha * image + beta

# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# axes = axes.ravel()

# images = [image, added_image, negative_image, multiplied_image, power_transformed_image, log_transformed_image, hist, cdf_normalized, contrast_image]
# titles = ['Исходное изображение', 'Добавление константы', 'Негатив', 'Умножение на константу', 'Степенное преобразование', 'Лог. преобразование', 'Гистограмма', 'Экв. гистограмма', 'Линейное контрастирование']

# for i in range(9):  # Now we have 9 plots
#     if i == 6:
#         axes[i].bar(range(256), hist, color='b', alpha=0.5)
#     elif i == 7:
#         axes[i].plot(cdf_normalized, color='b')
#         axes[i].hist(image.ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
#         axes[i].legend(('CDF', 'Гистограмма'), loc='upper left')
#     else:
#         axes[i].imshow(images[i], cmap='gray')
#     axes[i].set_title(titles[i])


# fig.delaxes(axes[9])

# plt.subplots_adjust(wspace=0.5, hspace=0.5)  
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(ax, image, title):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    ax.hist(image.flatten(), 256, [0, 256], color='r', alpha=1.0, density=True)
    ax.set_title(title)

def linear_contrast(image, alpha, beta):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result

def nonlinear_filter(image, filter_type, filter_size):
    if filter_type == 'median':
        result = cv2.medianBlur(image, filter_size)
    elif filter_type == 'maximum':
        result = cv2.dilate(image, np.ones((filter_size, filter_size), np.uint8))
    elif filter_type == 'minimum':
        result = cv2.erode(image, np.ones((filter_size, filter_size), np.uint8))
    else:
        raise ValueError("Unknown filter type")
    return result

# Загрузка изображения
image_path = '/Users/sonyands/Desktop/bsu/лаб пкг/pkg3/Снимок экрана.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Эквализация гистограммы
equ_hist = cv2.equalizeHist(image)

# Линейное контрастирование
alpha = 1.5
beta = 30
linear_contrasted = linear_contrast(image, alpha, beta)

# Нелинейные фильтры
filter_size = 5

# Создание 2x4 полотна
fig, axs = plt.subplots(2, 4, figsize=(15, 10))

# Отображение оригинального изображения и его гистограммы
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].axis('off')
plot_histogram(axs[0, 1], image, 'Original Histogram')

# Эквализированная гистограмма
plot_histogram(axs[0, 2], equ_hist, 'Equalized Histogram')

# Линейное контрастирование
axs[1, 0].imshow(linear_contrasted, cmap='gray')
axs[1, 0].axis('off')
axs[1, 0].set_title('Linear Contrast')

# Нелинейные фильтры: медианный, максимум, минимум
nonlinear_filtered_median = nonlinear_filter(image, 'median', filter_size)
nonlinear_filtered_maximum = nonlinear_filter(image, 'maximum', filter_size)
nonlinear_filtered_minimum = nonlinear_filter(image, 'minimum', filter_size)

axs[1, 1].imshow(nonlinear_filtered_median, cmap='gray')
axs[1, 1].axis('off')
axs[1, 1].set_title('Median Filter')

axs[1, 2].imshow(nonlinear_filtered_maximum, cmap='gray')
axs[1, 2].axis('off')
axs[1, 2].set_title('Maximum Filter')

axs[1, 3].imshow(nonlinear_filtered_minimum, cmap='gray')
axs[1, 3].axis('off')
axs[1, 3].set_title('Minimum Filter')

# Удаление лишних пустых графиков
for ax in axs[0, 3:]:
    ax.axis('off')
for ax in axs[1, 4:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt

impath = "image.jpg"
img = cv2.imread(impath)
#1полутон
image_g = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
cv2.imwrite("image_g.jpg", image_g)

#2логарифмическое преобразование

def log(img):
    img = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    log_img = np.uint8(log_img)
    return log_img

log_img = log(image_g)

cv2.imwrite("log_img.jpg", log_img)

#3 степенное преобразование

def power(image, power):
    img = image / 255.0
    result = np.power(img, power)
    result = np.uint8(result * 255)
    return result

power01 = power(image_g, 0.1)
power045 = power(image_g, 0.45)
power5 = power(image_g, 5)

cv2.imwrite('power_01.jpg', power01)
cv2.imwrite('power_045.jpg', power045)
cv2.imwrite('power_5.jpg', power5)

#4 соль и перец

def salt_pepper_noise(img, prob):
    noisy = np.copy(img)

    rnd = np.random.rand(img.shape[0], img.shape[1])

    noisy[rnd < prob / 2] = 0
    noisy[rnd > 1 - prob / 2] = 255

    return noisy

salt_papper = salt_pepper_noise(image_g, 0.05)

cv2.imwrite("salt_pepper.jpg", salt_papper)

plt.hist(salt_papper.ravel(), bins=256, range=(0,256))

plt.title("Гистограмма соль и перец")
plt.xlabel("Интенсивность пикселей")
plt.ylabel("Частота")
plt.savefig("hist_salt_pepper.jpg")

#5 Пространственная фильтрация зашумленного с медианным фильтром

mask = [3, 9, 15]

for size in mask:
    filtered = cv2.medianBlur(salt_papper, size)
    cv2.imwrite(f"filtered_median_{size}.png", filtered)

#6фильтр повышения резкости(исходного изображения)

kernel = np.array(
            [[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

sharpened = cv2.filter2D(image_g, -1, kernel)

cv2.imwrite("sharpened.jpg", sharpened)

#7Робертса

kernel_roberts_x = (np.array
                    ([[1, 0],
                             [0, -1]], dtype=np.float32))
kernel_roberts_y = (np.array
                    ([[0, 1],
                             [-1, 0]], dtype=np.float32))

edge_roberts_x = cv2.filter2D(image_g, -1, kernel_roberts_x)
edge_roberts_y = cv2.filter2D(image_g, -1, kernel_roberts_y)

edge_roberts = cv2.addWeighted(edge_roberts_x, 0.5, edge_roberts_y, 0.5, 0)

cv2.imwrite("edge_roberts.jpg", edge_roberts)

#Преввита

kernel_prewitt_x = (np.array
                    ([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32))
kernel_prewitt_y = (np.array
                    ([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], dtype=np.float32))

edge_prewitt_x = cv2.filter2D(image_g, -1, kernel_prewitt_x)
edge_prewitt_y = cv2.filter2D(image_g, -1, kernel_prewitt_y)

edge_prewitt = cv2.addWeighted(edge_prewitt_x, 0.5, edge_prewitt_y, 0.5, 0)

cv2.imwrite("edge_prewitt.jpg", edge_prewitt)

#Собеля

edge_sobel_x = cv2.Sobel(image_g, cv2.CV_64F, 1, 0, ksize=3)
edge_sobel_y = cv2.Sobel(image_g, cv2.CV_64F, 0, 1, ksize=3)

edge_sobel_x = cv2.convertScaleAbs(edge_sobel_x)
edge_sobel_y = cv2.convertScaleAbs(edge_sobel_y)

edge_sobel = cv2.addWeighted(edge_sobel_x, 0.5, edge_sobel_y, 0.5, 0)
cv2.imwrite("edge_sobel.jpg", edge_sobel)

#8 (ДОП)Нерезкое маскирование

def unsharp_mask(image, ksize=(5, 5), sigma=1.0, amount=1.5):

    blurred = cv2.GaussianBlur(image, ksize, sigma)

    mask = cv2.subtract(image, blurred)

    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)

    return sharpened


unsharp = unsharp_mask(image_g, (5, 5), 1.0, 1.5)

cv2.imwrite("unsharp.jpg", unsharp)

comparison = np.hstack((image_g, sharpened, unsharp))

cv2.imwrite("comparison_sharpening.jpg", comparison)

plt.figure(figsize=(10, 5))
plt.imshow(comparison, cmap='gray')
plt.title("Оригинал | Ядро резкости | Нерезкое маскирование")
plt.axis('off')
plt.savefig("comparison_plot.jpg")
plt.show()

def add_strip(image):
    img = image.copy()

    img[3, :] = 0

    return img

def filtr(image, ksize):

    filtred= cv2.medianBlur(image, ksize)

    return filtred

strip_img = add_strip(image_g)

filtered_strip = filtr(strip_img, 3)

cv2.imwrite("strip_image.jpg", strip_img)
cv2.imwrite("filtered_strip.jpg", filtered_strip)
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/mnt/data/eae5b13b-2e27-46e0-beaa-9240cfafc201.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pixels = img.reshape((-1, 3)).astype(np.float32)

K = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
quantized = centers[labels.flatten()]
quantized_img = quantized.reshape(img.shape)

plt.imshow(quantized_img)
plt.axis("off")
plt.show()

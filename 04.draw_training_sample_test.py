import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

_HEIGHT = 200
_WIDTH = 200


img = np.zeros((_HEIGHT, _WIDTH), np.uint8)
cv2.circle(img, (100, 150), 20, (255,255,255), 3)
cv2.rectangle(img, (50, 50), (70, 70), (255,255,255), -1)
cv2.imshow('hello', img)
cv2.waitKey(0)
cv2.destroyWindow()

cv2.imwrite('img.png', img)

im = np.array(imageio.imread('img.png'))
print(im.shape)
plt.imshow(im, interpolation='none', cmap='Blues')
plt.show()
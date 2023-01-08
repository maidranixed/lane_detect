import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurr = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurr, 50, 150)
# cv2.imshow("res", canny)
# cv2.waitKey(0)
plt.imshow(canny)
plt.show()

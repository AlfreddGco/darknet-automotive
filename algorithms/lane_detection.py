import cv2
import numpy as np
from matplotlib import pyplot as plt
import time, math

img = cv2.imread('./road.png')

start = time.time()

img = img[img.shape[0]//2:, :]

img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

edges = cv2.Canny(img, 100, 200, None, 3)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge image')
plt.xticks([])
plt.yticks([])


cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    

print('{:.2f} seconds'.format(time.time() - start))

cv2.imshow("Detected Lines (in red) - Line Transform", cdst)

plt.show()

cv2.waitKey(0)

import numpy as np
import cv2

# height, width = 100, 200
height, width = 5, 10
mat = np.zeros((height, width, 3), dtype=np.uint8)

mat[:, :, 0] = 3
mat[:, :, 1] = 3
mat[:, :, 2] = 100

cv2.imwrite("sample.jpg", mat)
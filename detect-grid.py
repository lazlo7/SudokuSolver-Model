import cv2
import numpy as np
from timeit import default_timer as timer

MIN_CELL_CONTOUR_AREA = 500
MAX_CELL_CONTOUR_AREA = 1200
CELL_MARGIN = 1

image = cv2.imread("sudoku.png")
start = timer()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Detect the grid's main border by contour area.
main_contour = max(contours, key=lambda contour: cv2.contourArea(contour))

# Mask out eveything outside of the grid's main border.
mask = np.zeros((gray.shape), np.uint8)
cv2.drawContours(mask, [main_contour], 0, 255, -1)
cv2.drawContours(mask, [main_contour], 0, 0, 2)
masked_image = np.zeros_like(gray)
masked_image[mask == 255] = gray[mask == 255]

# Detect the grid's inner cells. 
blur = cv2.bilateralFilter(masked_image, 8, 75, 75)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
good_contours = list(filter(lambda contour: MIN_CELL_CONTOUR_AREA < cv2.contourArea(contour) < MAX_CELL_CONTOUR_AREA, contours))
end = timer()
print(len(good_contours))
print(f"Frame took {end - start}s")

for contour_idx, contour in enumerate(good_contours):
    p1 = contour.min(axis=0)[0]
    p2 = contour.max(axis=0)[0]
    #cell = masked_image[y1 + CELL_MARGIN:y2 - CELL_MARGIN, x1 + CELL_MARGIN:x2 - CELL_MARGIN]
    image = cv2.rectangle(image, p1, p2, (0, 255, 0), 1)
    #cv2.imshow(f"Image #{contour_idx}", cell)
    #cv2.waitKey(500)
    #cv2.destroyAllWindows()

cv2.imshow(f"Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
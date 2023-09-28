import cv2
import numpy as np
import sys
import torch
from typing import Sequence
from train import BigConvNet
from functools import cmp_to_key
from torchvision.transforms import functional as F

MIN_CELL_CONTOUR_AREA = 500
MAX_CELL_CONTOUR_AREA = 1200
CELL_MARGIN = 3
CELL_HEIGHTS_MAX_DIFFERENCE = 12
CELLS_IN_ROW = 9
CELL_MAX_SIZE = 28

def detect_contours(grayscaled_image: cv2.UMat) -> Sequence[cv2.UMat]:
    blur = cv2.GaussianBlur(grayscaled_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_cell_image(image: cv2.UMat, top_left: np.ndarray, bottom_right: np.ndarray) -> cv2.UMat | None:
    dy = bottom_right[0] - top_left[0]
    dx = bottom_right[1] - top_left[1]

    res = np.full((28, 28), -1.0, np.float32)
    max_value = 0

    for y in range(CELL_MARGIN, dy - CELL_MARGIN):
        for x in range(CELL_MARGIN, dx - CELL_MARGIN):
            v = image[top_left[0] + y, top_left[1] + x]
            # Normalize to range (0, 1).
            v /= 255
            # Remap to range (1, 0)
            v = 1 - v
            if v < 0.6:
                v = 0
            max_value = max(max_value, v)
            # Convert to pytorch's model range (-1, 1).
            v = 2 * (v - 0.5)
            res[y, x] = v

    # Return None for empty cells.
    if max_value < 0.6:
        return None

    return res

def detect_grid_numbers(file_path: str) -> list[np.matrix] | None:
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the grid's main border by contour area.
    contours = detect_contours(gray)
    main_contour = max(contours, key=lambda contour: cv2.contourArea(contour))

    # Mask out eveything outside of the grid's main border.
    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [main_contour], 0, 255, -1)
    cv2.drawContours(mask, [main_contour], 0, 0, 2)
    masked_image = np.zeros_like(gray)
    masked_image[mask == 255] = gray[mask == 255]

    # Detect the grid's inner cells. 
    contours = detect_contours(masked_image)
    contours = list(filter(lambda contour: MIN_CELL_CONTOUR_AREA < cv2.contourArea(contour) < MAX_CELL_CONTOUR_AREA, contours))

    # There should always be 81 contours (=81 grid cells)
    if len(contours) != 81:
        return None

    # Find corner points of each detected cell contour.
    # A tuple of 2 corner points represents a cell grid of size 28x28 (=CELL_MAX_SIZExCELL_MAX_SIZE)  
    corner_ps = []
    for contour in contours:
        top_left = contour.min(axis=0)[0][::-1]
        bottom_right = contour.max(axis=0)[0][::-1]
        
        center = (top_left + bottom_right) // 2 

        top_left_adjusted = np.maximum(top_left, center - CELL_MAX_SIZE // 2)
        bottom_right_adjusted = np.minimum(bottom_right, center + CELL_MAX_SIZE // 2)
        corner_ps.append((top_left_adjusted, bottom_right_adjusted))

    def compare(ps1: np.ndarray, ps2: np.ndarray):
        if ps1[0][0] < ps2[0][0]:
            return -1
        if ps1[0][0] > ps2[0][0]:
            return 1
        return ps1[0][1] - ps2[0][1]

    # Detect rows by the top-left corner point.
    corner_ps.sort(key=cmp_to_key(compare))
    print(len(corner_ps))

    res = []
    row_ps = []
    last_row_y = corner_ps[0][0][0]
    detected_in_row = 0

    for ps in corner_ps:
        top_left_y = ps[0][0] 
 
        if detected_in_row % CELLS_IN_ROW == 0:
            last_row_y = top_left_y
            res.extend(sorted(row_ps, key=lambda ps: ps[0][1]))
            row_ps.clear()

        detected_in_row += 1       

        # The difference between two consecutive cell heights is too large - image is too skewed.
        if abs(top_left_y - last_row_y) > CELL_HEIGHTS_MAX_DIFFERENCE:
            return None

        row_ps.append(ps)

    res.extend(sorted(row_ps, key=lambda ps: ps[0][1]))
    return [ get_cell_image(masked_image, ps[0], ps[1]) for ps in res ]

def print_number(image: np.ndarray):
    mask = "@=-."
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_norm = 0.5 * (-image[i, j] + 1)
            print(mask[round(255 * image_norm) // (256 // len(mask))], end="")
        print()

def print_grid(numbers: list[int]):
    for i in range(9):
        for j in range(9):
            v = numbers[i*9 + j]
            print(v if v is not None else ".", end=" ")
        print()
        
def forward(model_file_path: str, input_file_path: str):
    print("Parsing image...")
    input_numbers = detect_grid_numbers(input_file_path)

    print("Loading model...")
    model = BigConvNet()
    model.cuda()
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    res = []
    print("Forwarding...")
    with torch.no_grad():
        for input_number in input_numbers:
            # Empty cell.
            if input_number is None:
                res.append(None)
                continue
            out = model(torch.from_numpy(input_number)).cpu().numpy()
            recognised_number = out.argmax()
            res.append(recognised_number)

    print_grid(res)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <mode_file_path> <input_file_path>")
    else:
        forward(sys.argv[1], sys.argv[2])
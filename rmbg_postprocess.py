import numpy as np
import cv2

def apply_sigmoid(x, exp=1.06, slope=135, value_range=255.0):
    x = x.astype(np.float64)
    y = 1/(1+exp**(slope-x))*value_range
    return y.astype(np.uint8)

def apply_open(mask, erode_iter=2, dia_iter=2):
    kernel_size = max(3, min(mask.shape[0]//256, mask.shape[1]//256))
    erode_mask = cv2.erode(mask, np.ones((kernel_size,kernel_size),np.uint8), iterations=erode_iter)
    dia_mask = cv2.dilate(erode_mask, np.ones((kernel_size,kernel_size),np.uint8), iterations=dia_iter)
    return dia_mask


if __name__ == "__main__":
    mask = np.ones((800,600,3)) * 255
    mask_sigmoid = apply_sigmoid(mask)
    mask_open = apply_open(mask_sigmoid)
    print("mask:", mask_open)
    print("mask shape:", mask.shape)
    print("processed mask shape:", mask_open.shape)
    
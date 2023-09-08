import numpy as np
import cv2

# 이미지 후처리 클래스
class MaskPostProcessor:
    def __init__(self):
        self.kernel_size = 3

    # cv2 change image type
    def ch_GRAY2BGR(self, gray_img):
        bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        return bgr_img    
    def ch_BGR2GRAY(self, bgr_img):
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        return gray_img
    def ch_BGR2RGBA(self, bgr_img):
        rgba_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGBA)
        return rgba_img


    # 커널 사이즈 조정
    def set_kernel_size(self, mask_shape):
        self.kernel_size = max(3, min(mask_shape[0]//256, mask_shape[1]//256))
    

    # cv2 erode dilate 함수
    def apply_erode(self, mask, iteration=2):
        erode_mask = cv2.erode(mask, np.ones((self.kernel_size,self.kernel_size),np.uint8), iterations=iteration)
        return erode_mask
    
    def apply_dilate(self, mask, iteration=2):
        dia_mask = cv2.dilate(mask, np.ones((self.kernel_size,self.kernel_size),np.uint8), iterations=iteration)
        return dia_mask
    
    def apply_open(self, mask, erode_iter=2, dia_iter=2):
        erode_mask = self.apply_erode(mask, iteration=erode_iter)
        dia_mask = self.apply_dilate(erode_mask, iteration=dia_iter)
        return dia_mask

    def apply_close(self, mask, erode_iter=2, dia_iter=2):
        dia_mask = self.apply_dilate(mask, iteration=dia_iter)
        erode_mask = self.apply_erode(dia_mask, iteration=erode_iter)
        return erode_mask
    

    # sigmoid 픽셀 값 조정
    def apply_sigmoid(self, x, exp=1.06, slope=135, value_range=255.0):
        x = x.astype(np.float64)
        y = 1/(1+exp**(slope-x))*value_range
        return y.astype(np.uint8)
    

    # 가장 큰 contour 영역 마스크로 만들어서 반환
    def get_largest_contour_mask(self, mask):
        contour_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return contour_mask


    # 커스텀 함수
    def apply_custom_process(self, mask):
        '''For Remove background
        input: mask matrix(BGR or Gray) / output: mask matrix(BGR)

        마스크 => sigmoid_bg, sigmoid_obj 두번 적용=>
        sigmoid_bg에 대해서 가장 큰 contour영역 마스킹 => 
        마스킹 erode(obj 경계선 줄이기) =>
        contour의 bg, obj 영역에 다른 sigmoid 결과를 할당 => 
        erode, close 적용 => 결과 리턴'''

        if len(mask.shape) != 2:
            mask = self.ch_BGR2GRAY(mask)
        self.set_kernel_size(mask.shape)

        mask_sig_bg = self.apply_sigmoid(mask, exp=1.2, slope=135, value_range=255)
        mask_sig_obj = self.apply_sigmoid(mask, exp=1.06, slope=60, value_range=255)

        contour_mask = self.get_largest_contour_mask(mask_sig_bg)
        contour_mask = self.apply_erode(mask, iteration=3)

        mask_sig_contour = np.where(contour_mask == 0, mask_sig_bg, mask_sig_obj)

        result_mask = self.apply_open(mask_sig_contour, erode_iter=2, dia_iter=2)

        return self.ch_GRAY2BGR(result_mask)



if __name__ == "__main__":
    mask = (np.ones((800,600,3)) * 255).astype('uint8')

    postprocessor = MaskPostProcessor()
    result = postprocessor.apply_custom_process(mask)
    
    print("mask:", mask)
    print("mask shape:", mask.shape)
    print("processed mask shape:", result.shape)
    
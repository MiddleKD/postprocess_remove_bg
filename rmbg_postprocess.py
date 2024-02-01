import numpy as np
import cv2
from PIL import Image

def resize_image(image, height, width):
    # cv2의 resize는 width, height 순서로 전달해야 함
    resized_image = cv2.resize(image, [width, height], interpolation=cv2.INTER_AREA)
    return resized_image

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
    def bgr_to_pil(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img.astype(np.uint8))



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
    def apply_sigmoid(self, x, exp=1.06, slope=135, value_range=255.0, normalize=False):
        x = x.astype(np.float64)
        y = 1/(1+exp**(slope-x))*value_range
        result = y

        if normalize == True:
            min_val = 1/(1+exp**(slope-0))*value_range
            max_val = 1/(1+exp**(slope-255))*value_range
            result = (y - min_val) / (max_val - min_val)*value_range

        return result.astype(np.uint8)
    

    # 가장 큰 contour 영역 마스크로 만들어서 반환
    def get_largest_contour_mask(self, mask):
        contour_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(np.where(np.array(mask)>127,255,0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_mask, [largest_contour], 0, 1, thickness=cv2.FILLED)
        return contour_mask


    # 커스텀 함수
    def apply_2_sigmoid_region_process(self, mask):
        '''For Remove background
        input: mask matrix(BGR or Gray) / output: mask matrix(BGR)

        마스크 => sigmoid_bg, sigmoid_obj 두번 적용=>
        sigmoid_bg에 대해서 가장 큰 contour영역 마스킹 => 
        마스킹 erode(obj 경계선 축소) =>
        contour의 bg, obj 영역에 다른 sigmoid 결과를 할당 =>
        결과 리턴'''

        if len(mask.shape) != 2:
            mask = self.ch_BGR2GRAY(mask)
        self.set_kernel_size(mask.shape)

        mask_sig_bg = mask
        # mask_sig_bg = self.apply_sigmoid(mask, exp=1.06, slope=135, value_range=255, normalize=True)
        mask_sig_obj = self.apply_sigmoid(mask, exp=1.06, slope=60, value_range=255, normalize=True)
        
        contour_mask = self.get_largest_contour_mask(mask_sig_bg)
        contour_mask = self.apply_erode(contour_mask, iteration=5)

        mask_sig_contour = np.where(contour_mask == 0, mask_sig_bg, mask_sig_obj)
        
        result_mask = mask_sig_contour
        # result_mask = self.apply_open(mask_sig_contour, erode_iter=2, dia_iter=2)

        return self.ch_GRAY2BGR(result_mask)
    
    # 커스텀 함수2
    def apply_3_sigmoid_region_process(self, mask):
        '''For Remove background
        input: mask matrix(BGR or Gray) / output: mask matrix(BGR)

        마스크 => sigmoid_bg, sigmoid_obj_erode, sigmoid_obj_dilate 세번 적용=>
        sigmoid_bg에 대해서 가장 큰 contour영역 마스킹 => 
        마스킹 erode(obj 경계선 축소), 마스킹 dilate(obj 경계선 확장) =>
        contour의 bg, obj(축소), obj(확장) 영역에 다른 sigmoid 결과를 할당 =>
        결과 리턴'''

        if len(mask.shape) != 2:
            mask = self.ch_BGR2GRAY(mask)
        self.set_kernel_size(mask.shape)

        mask_sig_bg = self.apply_sigmoid(mask, exp=1.02, slope=150, value_range=255, normalize=True)
        mask_sig_obj_erode = self.apply_sigmoid(mask, exp=1.06, slope=60, value_range=255, normalize=True)
        mask_sig_obj_dilate = mask
        
        contour_mask = self.get_largest_contour_mask(mask_sig_bg)
        contour_mask_erode = self.apply_erode(contour_mask, iteration=5)
        contour_mask_dialte = self.apply_dilate(contour_mask, iteration=3)

        mask_sig_contour = np.where(contour_mask_erode == 1, mask_sig_obj_erode, np.where(contour_mask_dialte == 1, mask_sig_obj_dilate, mask_sig_bg))
        
        result_mask = mask_sig_contour
        # result_mask = self.apply_open(mask_sig_contour, erode_iter=2, dia_iter=2)

        return self.ch_GRAY2BGR(result_mask)
    
    def __call__(self, mask_pil):
        mask = np.array(mask_pil)
        mask_bgr = self.apply_3_sigmoid_region_process(mask)
        mask_pil = self.bgr_to_pil(mask_bgr)
        return mask_pil


def visualize_green(img, mask):
    import matplotlib.pyplot as plt
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bg = ((1- mask/255.0)*[120,255,155]).astype(np.uint8)
    temp = (np.array(img)*(mask/255.0)).astype(np.uint8)
    mask_pil = Image.fromarray(bg + temp)
    plt.imshow(mask_pil)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    from transparent_background import Remover
    remover = Remover(fast=True, device="cuda")

    img = Image.open("./test_img/test1.jpg")
    
    mask = remover.process(img, type="map")

    postprocessor = MaskPostProcessor()
    mask_processed = postprocessor(mask)
    mask_processed.show()
    
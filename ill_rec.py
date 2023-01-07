import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance

class IllCorr(nn.Module):
    def __init__(self, ks=20, enhance=True):
        super().__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=ks, padding=ks//2, stride=1)
        self.enhance = enhance

    def background_subtraction(self, I: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        O = (I - B).squeeze(0).numpy()
        norm_img = cv2.normalize(O, None, 0,255, norm_type=cv2.NORM_MINMAX)
        return norm_img
    
    def enhance_image(self, img) -> np.ndarray:
        img = Image.fromarray(img).convert('RGB')
        enhance_brightness = ImageEnhance.Brightness(img)
        img = enhance_brightness.enhance(factor=1.4)
        enhance_contrast = ImageEnhance.Contrast(img)
        img = enhance_contrast.enhance(factor=1.7)
        # enhance_sharpness = ImageEnhance.Sharpness(img)
        # img = enhance_sharpness.enhance(factor=1.1)
        return np.array(img.convert('F'))

    def forward(self, im: torch.Tensor) -> np.ndarray:
        # print(im.shape)
        a = self.max_filter(im)
        # print(a.shape)
        b = (-self.max_filter(-a))[:,:-2,:-2]
        # print(b.shape)
        out = self.background_subtraction(im, b)
        if self.enhance:
            out = self.enhance_image(out)

        return out

def rec_ill(img, saveRecPath):
    net = IllCorr(ks=20, enhance=True)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    out = net(torch.Tensor(img).unsqueeze(0))
    cv2.imwrite(saveRecPath, out)

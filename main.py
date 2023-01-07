import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from seg import U2NETP
from GeoTr import GeoTr
from ill_rec import rec_ill

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        print('Sementation working...', end='')
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x
        print('Done.')
        print('GeoTr working...', end='')
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm

def reload_model(model: GeoTr_Seg, path='./model_pretrained/') -> GeoTr_Seg:
    seg_model_dict = model.msk.state_dict()
    seg_pretrained_dict = torch.load(path + 'seg.pth', map_location='cpu')
    # print(len(seg_pretrained_dict.keys()))
    print('Segmentation model successfully reloaded.')
    seg_pretrained_dict = {k[6:]: v for k, v in seg_pretrained_dict.items() if k[6:] in seg_model_dict}
    # print(len(seg_pretrained_dict.keys()))
    seg_model_dict.update(seg_pretrained_dict)
    model.msk.load_state_dict(seg_model_dict)

    geo_model_dict = model.GeoTr.state_dict()
    geo_pretrained_dict = torch.load(path + 'geotr.pth', map_location='cpu')
    # print(len(geo_pretrained_dict.keys()))
    print('GeoTr model successfully reloaded.')
    geo_pretrained_dict = {k[7:]: v for k, v in geo_pretrained_dict.items() if k[7:] in geo_model_dict}
    # print(len(geo_pretrained_dict.keys()))
    geo_model_dict.update(geo_pretrained_dict)
    model.GeoTr.load_state_dict(geo_model_dict)

    return model

def rectify(options: argparse.Namespace):
    image_list = os.listdir(options.distorted_path)
    print(str(len(image_list)) + ' images to be process.')
    
    if not os.path.exists(options.geo_save_path):  # create save path
        os.mkdir(options.geo_save_path)
    if not os.path.exists(options.ill_save_path):  # create save path
        os.mkdir(options.ill_save_path)

    GeoTr_Seg_model = GeoTr_Seg().to(device=device)
    reload_model(GeoTr_Seg_model, options.model_path)
    
    GeoTr_Seg_model.eval()

    for image_path in image_list:
        name = image_path.split('.')[-2]
        image_path = options.distorted_path + image_path

        print('Begin: ', image_path)
        print('Reading image...', end='')
        image_original = np.array(Image.open(image_path))[:, :, :3] / 255.
        print('Done. Image resolution: ', image_original.shape)
        h, w, _ = image_original.shape

        image = cv2.resize(image_original, (288, 288))
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float().unsqueeze(0)

        with torch.no_grad():
            bm = GeoTr_Seg_model(image.to(device=device))

            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            
            out = F.grid_sample(torch.from_numpy(image_original).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)

            image_geo = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)
            print('Done.')
            print('Saving geo images...', end='')
            cv2.imwrite(options.geo_save_path + name + '_geo' + '.png', image_geo)
            print('Done.')

        if options.ill_rec:
            ill_save_path = options.ill_save_path + name + '_ill' + '.png'
            print('Illumination correction working...', end='')
            rec_ill(image_geo, ill_save_path)
            print('Done.')
        
        print('Done: ', image_path + '\n')
    print('All done!')

# 程序总入口
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distorted_path', default='./distorted/')    # 存放扭曲图片的源文件夹
    parser.add_argument('--geo_save_path',  default='./geo_rec/')          # 存放几何矫正输出图片的文件夹
    parser.add_argument('--ill_save_path',  default='./ill_rec/')          # 存放光照修复输出图片的文件夹
    parser.add_argument('--model_path',  default='./model_pretrained/')        # 存放边界分割训练模型的位置
    parser.add_argument('--ill_rec',  default=True)    # 是否进行光照修复，默认为是
    rectify(parser.parse_args())


if __name__ == '__main__':
    main()

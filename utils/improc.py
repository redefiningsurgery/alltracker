import torch
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import (
    rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
    rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)

def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform

hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)

def flow2color(flow, clip=0.0):
    B, C, H, W = list(flow.size())
    assert(C==2)
    flow = flow[0:1].detach()
    if clip==0:
        clip = torch.max(torch.abs(flow)).item()
    flow = torch.clamp(flow, -clip, clip)/clip
    radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) # B,1,H,W
    radius_clipped = torch.clamp(radius, 0.0, 1.0)
    angle = torch.atan2(-flow[:, 1:2], -flow[:, 0:1]) / np.pi # B,1,H,W
    hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
    saturation = torch.ones_like(hue) * 0.75
    value = radius_clipped
    hsv = torch.cat([hue, saturation, value], dim=1) # B,3,H,W
    flow = hsv_to_rgb(hsv)
    flow = (flow*255.0).type(torch.ByteTensor)
    return flow


COLORMAP_FILE = "./utils/bremm.png"
class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        self._img = (plt.imread(self._colormap_file)*255).astype(np.uint8)
        
        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3), dtype=np.uint8)
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int((self._width-1) * x)
            yp = int((self._height-1) * y)
            xp = np.clip(xp, 0, self._width-1)
            yp = np.clip(yp, 0, self._height-1)
            output[i, :] = self._img[yp, xp]
        return output

def get_2d_colors(xys, H, W):
    N,D = xys.shape
    assert(D==2)
    bremm = ColorMap2d()
    xys[:,0] /= float(W-1)
    xys[:,1] /= float(H-1)
    colors = bremm(xys)
    # print('colors', colors)
    # colors = (colors[0]*255).astype(np.uint8) 
    # colors = (int(colors[0]),int(colors[1]),int(colors[2]))
    return colors
    
    
def get_n_colors(N, sequential=False):
    label_colors = []
    for ii in range(N):
        if sequential:
            rgb = cm.winter(ii/(N-1))
            rgb = (np.array(rgb) * 255).astype(np.uint8)[:3]
        else:
            rgb = np.zeros(3)
            while np.sum(rgb) < 128: # ensure min brightness
                rgb = np.random.randint(0,256,3)
        label_colors.append(rgb)
    return label_colors

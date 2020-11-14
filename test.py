import torch
import os
from tqdm import tqdm
import numpy as np
from convnet import InpaintGenerator
import cv2
import argparse



parser = argparse.ArgumentParser(description='test script')
parser.add_argument('--nogpu', action='store_true')
parser.add_argument('--image', default='./examples/images', type=str)
parser.add_argument('--mask', default='./examples/masks', type=str)
parser.add_argument('--output', default='./examples/results', type=str)
args = parser.parse_args()


use_gpu = not args.nogpu
device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
net = InpaintGenerator()
net.load_state_dict(torch.load("./files/model_256.pth"))
net = net.to(device)

path_in = args.image
path_mask = args.mask
path_out = args.output

if not os.path.exists(path_out):
    os.mkdir(path_out)


for name in tqdm(os.listdir(path_mask), desc="processing"):
    img = cv2.imread(f"{path_in}/{name}", cv2.IMREAD_COLOR)
    img = img[:,:,::-1].copy()
    h_raw, w_raw, _ = img.shape
    img_raw = img

    img = torch.Tensor(img.transpose((2,0,1)))[None,...]/255.0
    img = (img-0.5)/0.5

    msk = cv2.imread(f"{path_mask}/{name}", 0)
    msk = cv2.resize(msk, (w_raw, h_raw))
    msk = (msk>0)[...,None]
    msk_raw = msk
    msk = torch.Tensor(msk.transpose((2,0,1)))[None,...]

    img = img.to(device)
    msk = msk.to(device)

    with torch.no_grad():
        _, result = net(img*(1-msk), msk)
        result = result*msk+img*(1-msk)
        result = result*0.5+0.5
    result = result.detach().cpu()[0].numpy()*255
    result = result.transpose((1,2,0)).astype(np.uint8)
    result = cv2.resize(result, (w_raw, h_raw))
    result = result*msk_raw + img_raw*(1-msk_raw)
    result = result[:,:,::-1]
    cv2.imwrite(f"{path_out}/{name}", result)


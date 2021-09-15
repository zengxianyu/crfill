import numpy as np
import cv2
import torch
import data
from options.test_options import TestOptions
import models

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

# test
num = 0
psnr_total = 0
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    with torch.no_grad():
        generated,_ = model(data_i, mode='inference')
    generated = torch.clamp(generated, -1, 1)
    generated = (generated+1)/2*255
    generated = generated.cpu().numpy().astype(np.uint8)
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        pred_im = generated[b].transpose((1,2,0))
        print('process image... %s' % img_path[b])
        cv2.imwrite(img_path[b], pred_im[:,:,::-1])

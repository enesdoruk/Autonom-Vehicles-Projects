import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from LightDetect.models.experimental import attempt_load
from LightDetect.utils.datasets import LoadStreams, LoadImages, letterbox
from LightDetect.utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from LightDetect.utils.plots import plot_one_box
from LightDetect.utils.torch_utils import select_device, load_classifier, time_synchronized


def light_detect(frame, save_img=False):
    weights, view_img,  imgsz = '../BTUCARF/LightDetect/best.pt', 1, 640
    
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  

    model = attempt_load(weights, map_location=device)  
    imgsz = check_img_size(imgsz, s=model.stride.max()) 
    if half:
        model.half()  
    
    names = model.module.names if hasattr(model, 'module') else model.names
    #print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 

    img = letterbox(frame, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0 

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    t1 = time_synchronized()
    pred = model(img, augment=True)[0]

    pred = non_max_suppression(pred, 0.7, 0.8, agnostic=True)
    t2 = time_synchronized()

    return img,pred,names,colors

    
    


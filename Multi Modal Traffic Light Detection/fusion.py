import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import warnings
warnings.filterwarnings("ignore")
import datetime
from time import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import cv2

import time

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms as NMS
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from ensemble_boxes import *


def light_detect(frame, save_img=False):
    weights, view_img,  imgsz = 'best.pt', 1, 960
    
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  

    model = attempt_load(weights, map_location=device)  
    imgsz = check_img_size(imgsz, s=model.stride.max()) 
    if half:
        model.half()  
    
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

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

    pred = model(img, augment=True)[0]

    pred = non_max_suppression(pred, 0.8, 0.9, agnostic=True)

    return img,pred,names,colors
    
def filterBoxes(output,nms_th=0.7,score_threshold=0.9):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    mask = NMS(boxes,scores,nms_th)
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels
    
def displayPredictions(output,nms_th=0.7,score_threshold=0.9):
    
    boxes,scores,labels = filterBoxes(output,nms_th,score_threshold)
    
    image = cv2.imread("resim.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image, (960,960))
    image /= 255.0
    
    
    for box,label in zip(boxes,labels):
        image = cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0,0,255), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (480,480))
    cv2.imshow("rcnn_result", image)
    cv2.waitKey(10000)

def yolo_run():
    RGB = cv2.imread("resim.jpg")
    RGB = cv2.resize(RGB, (960,960))
    boxes = []
    scores = []
    labels = []
    img,pred,names,colors = light_detect(RGB)
    for i, det in enumerate(pred):
        s, im0 =  '', RGB
        s += '%gx%g ' % img.shape[2:]  
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det): 
                labels.append(int(cls.item()))
                boxes.append(xyxy)
                scores.append(conf.item())
                plot_one_box(xyxy, im0, color=(255,0,0), line_thickness=1)
    
    im0 = cv2.resize(im0, (480,480))
    cv2.imshow("yolo_result", im0)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    
    return boxes, scores, labels
    
    
def rcnn_run():
    preprocess = transforms.Compose([
    transforms.ToTensor()])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = select_device('0')
    
    N_CLASS = 4  
    INP_FEATURES = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(INP_FEATURES, N_CLASS)

    model.load_state_dict(torch.load('rcnn.pth'))
    model.eval()
    model.to(device)

    images = cv2.imread("resim.jpg")
    images = cv2.resize(images, (960,960))
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    images = preprocess(images).float()
    images = images.unsqueeze_(0)
    images = images.type(torch.cuda.FloatTensor)
    outputs = model(images)

    displayPredictions(outputs[0],0.7,0.9)
    return outputs[0]

def IOU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
	iou = interArea / float(boxAArea + boxBArea - interArea)
    
	return iou

def show_image(im, name='WeightedFusion'):
    im = cv2.resize(im, (480,480))
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(10000)

def show_boxes(boxes_list, scores_list, labels_list, image_size=960, names=''):
    thickness = 2
    image = cv2.imread("resim.jpg")
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            y1 = int(image_size * boxes_list[i][j][1])
            x2 = int(image_size * boxes_list[i][j][2])
            y2 = int(image_size * boxes_list[i][j][3])
            print(x1, y1, x2, y2 )
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), int(thickness * scores_list[i][j]))
    show_image(image, name=names)

if __name__=="__main__":
   
    image = cv2.imread("resim.jpg")
    image = cv2.resize(image, (960,960))
    
    image2 = cv2.imread("resim.jpg")
    image2 = cv2.resize(image2, (960,960))
    
    image3 = cv2.imread("resim.jpg")
    image3 = cv2.resize(image3, (960,960))
    
    labeled = np.array([[529, 179, 545, 227],
                        [631, 185, 647, 236],
                        [741, 212, 758, 262],
                        [867, 294, 884, 342]])
    labeled = labeled.astype(int)
    
    result = []
    com_boxes = []
    com_scores = []
    com_labels = []
    weights = [2, 1]

    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
               
    outputs = rcnn_run()
    rcnn_boxes,rcnn_scores,rcnn_labels = filterBoxes(outputs,nms_th=0.7,score_threshold=0.9)
    for box,label in zip(rcnn_boxes,rcnn_labels):
        image = cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0,0,255), 2)
    
    yolo_boxes, yolo_scores, yolo_labels = yolo_run()
    for i in yolo_boxes:  
        plot_one_box(i, image, color=(255,255,255), line_thickness=1)
    yolo_bx = []
    for i in range(len(yolo_boxes)):
        yolo_bx.append([yolo_boxes[i][0].item(),yolo_boxes[i][1].item(),yolo_boxes[i][2].item(),yolo_boxes[i][3].item()])
    
    
    for i in range(len(yolo_bx)):
        for j in range(len(rcnn_boxes)):
            #print("RCNN BOX = ", rcnn_boxes[j])
            #print("YOLO BOX = ", yolo_bx[i])
            #print("IOU = ", IOU(np.array(yolo_bx[i]), np.array(rcnn_boxes[j])))
            
            if IOU(np.array(yolo_bx[i]), np.array(rcnn_boxes[j])) > 0.5:
                if yolo_scores[i] > rcnn_scores[j]:
                    #print("YOLO SCORES = ", yolo_scores[i])
                    #print("RCNN SCORES = ", scores[j])
                    result.append(yolo_bx[i])
                if rcnn_scores[j] > yolo_scores[i]:
                    #print("YOLO SCORES = ", yolo_scores[i])
                    #print("RCNN SCORES = ", scores[j])
                    result.append(rcnn_boxes[j])
    
    
    rcnn_boxes = rcnn_boxes.tolist()
    rcnn_labels = rcnn_labels.tolist()
    rcnn_scores = rcnn_scores.tolist()
    
    for i in range(len(rcnn_boxes)):
        for j in range(len(rcnn_boxes[0])):
            rcnn_boxes[i][j] = rcnn_boxes[i][j] / 960
            
    for i in range(len(yolo_bx)):
        for j in range(len(yolo_bx[0])):
            yolo_bx[i][j] = yolo_bx[i][j] / 960
    
    com_boxes.append(yolo_bx)
    com_boxes.append(rcnn_boxes)
    com_scores.append(yolo_scores)
    com_scores.append(rcnn_scores)
    com_labels.append(yolo_labels)
    com_labels.append(rcnn_labels)
       
    
    boxes_nms, scores_nms, labels_nms = nms(com_boxes, com_scores, com_labels, weights=weights, iou_thr=iou_thr)
    boxes_softNMS, scores_softNMS, labels_softNMS = soft_nms(com_boxes, com_scores, com_labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    boxes_nmw, scores_nmw, labels_nmw = non_maximum_weighted(com_boxes, com_scores, com_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(com_boxes, com_scores, com_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    
    show_boxes([boxes_nms], [scores_nms], [labels_nms], image_size=960, names='NMS')
    show_boxes([boxes_softNMS], [scores_softNMS], [labels_softNMS], image_size=960, names='SoftNMS')
    show_boxes([boxes_nmw], [scores_nmw], [labels_nmw], image_size=960, names='NMW')
    show_boxes([boxes_wbf], [scores_wbf], [labels_wbf], image_size=960, names='WBF')
    
    
    for i in range(len(labeled)):
        for j in range(len(yolo_bx)):
            if IOU(np.array(labeled[i]),np.array(yolo_bx[j])) > 0:
                print("YOLO IOU = ",IOU(np.array(labeled[i]),np.array(yolo_bx[j])))
        for k in range(len(rcnn_boxes)):
            if IOU(np.array(labeled[i]),np.array(rcnn_boxes[k])) > 0:
                print("RCNN IOU = ",IOU(np.array(labeled[i]),np.array(rcnn_boxes[k])))
        for l in range(len(result)):
            if IOU(np.array(labeled[i]),np.array(result[l])) > 0:
                print("FUSION IOU = ",IOU(np.array(labeled[i]),np.array(result[l])))
    
    for i in range(len(result)):
        image2 = cv2.rectangle(image2, (int(result[i][0]), int(result[i][1])),
                              (int(result[i][2]), int(result[i][3])), (0,0,255), 1)
    for i in range(len(labeled)):
        image3 = cv2.rectangle(image3, (int(labeled[i][0]), int(labeled[i][1])),
                              (int(labeled[i][2]), int(labeled[i][3])), (0,0,255), 1)
    

    
    image3 = cv2.resize(image3, (480,480))
    cv2.imshow("labeled_result", image3)
    cv2.waitKey(10000)
    
    image2 = cv2.resize(image2, (480,480))
    cv2.imshow("fusion_result", image2)
    cv2.waitKey(10000)
    
    image = cv2.resize(image, (480,480))
    cv2.imshow("adding two model", image)
    cv2.waitKey(10000)    
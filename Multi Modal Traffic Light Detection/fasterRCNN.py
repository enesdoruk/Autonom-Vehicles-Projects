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

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler


def filterBoxes(output,nms_th=0.7,score_threshold=0.9):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    mask = nms(boxes,scores,nms_th)
    
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
    
    image = cv2.imread("light.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    #image = cv2.resize(image,(512,512))
    image /= 255.0
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    for box,label in zip(boxes,labels):
        print(box)
        image = cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0,0,255), 2)

    ax.set_axis_off()
    ax.imshow(image)

    plt.show()

preprocess = transforms.Compose([
    transforms.ToTensor()])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
N_CLASS = 4  
INP_FEATURES = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(INP_FEATURES, N_CLASS)

model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth'))
model.eval()

images = cv2.imread("light.jpg")
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
images = preprocess(images).float()
images = images.unsqueeze_(0)
images = images.type(torch.FloatTensor)
outputs = model(images)

print(outputs)
displayPredictions(outputs[0],0.7,0.9)
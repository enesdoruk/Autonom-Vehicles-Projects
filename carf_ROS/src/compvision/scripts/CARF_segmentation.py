#!/usr/bin/python
# -*- coding: UTF-8 -*-

import rospy    
import cv2
from sensor_msgs.msg import Image
import sys
import numpy as np
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int32MultiArray

import argparse
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image as pilImage
from torchvision import transforms
from scipy import ndimage as ndi
from skimage.color import label2rgb


from Segmentation.unet import UNet
from Segmentation.data_vis import plot_img_and_mask
from Segmentation.dataset import BasicDataset

import time

prev_frame_time = 0
new_frame_time = 0

SegmentTalker = rospy.Publisher('Segmentation', Image, queue_size=10)
StartStop = rospy.Publisher('StartStop', Int32MultiArray, queue_size=10)

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_args():
  
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return pilImage.fromarray((mask * 255).astype(np.uint8))

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding")
    dtype = np.dtype("uint8") 
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data) 
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height 

    return img_msg

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x = 0
    y = 0
    w = 0
    h = 0

    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h

    return x, y, w, h

def LaneSegmentation(RGB):
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model ")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net.load_state_dict(torch.load('/home/carf/carf_ROS/src/compvision/scripts/Segmentation/checkpoints/CP_epoch21.pth', map_location=device))

    #im = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)
    #RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(RGB, (640, 640))
    img = pilImage.fromarray(RGB)

    mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.5,
                           device=device)

    result = mask_to_image(mask)
    resultcv = np.array(result)
   
    '''
    for i in range(resultcv.shape[0]):
            
        y1 = 0
        y2 = 0
        y = 0

        for j in range(resultcv.shape[1]):
            if (resultcv[i][j] > 0):
                y1 = j
                break

        for k in range(resultcv.shape[1]-1):
            if (resultcv[i][(resultcv.shape[1] -1) - k] > 0):
                y2 = (resultcv.shape[1] -1) - k
                break

        if (abs(y1 - y2) > 10):
            y = (y1 + y2) // 2
            resultcv[i][y] = 255          
        '''
    '''
    inptcvRGB = cv2.resize(RGB, (640, 640))
    labeled_coins, _ = ndi.label(resultcv)
    image_label_overlay = label2rgb(labeled_coins, image=resultcv, bg_label=-1)
    image_label_overlay = image_label_overlay.astype('uint8') * 255
    image_label_overlay_gray = cv2.cvtColor(image_label_overlay, cv2.COLOR_BGR2GRAY)
       
    resultcvRGB = cv2.cvtColor(resultcv, cv2.COLOR_GRAY2RGB)
    resultcvRGB = cv2.resize(resultcvRGB, (640, 640))

    WCol1 = (255,255,255)
    WCol2= (50,50,50)
    mask = cv2.inRange(resultcvRGB, WCol2, WCol1)
    resultcvRGB[mask>0] = (0,0,255)
    
    combine = cv2.add(resultcvRGB, inptcvRGB)
    '''

    return resultcv



def startStopSegmentation(RGB):
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model ")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net.load_state_dict(torch.load('/home/carf/carf_ROS/src/compvision/scripts/Segmentation/checkpoints/CP_epoch25.pth', map_location=device))

    #im = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)
    #RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(RGB, (640, 640))
    img = pilImage.fromarray(RGB)

    mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=0.5,
                           out_threshold=0.8,
                           device=device)

    result = mask_to_image(mask)
    resultcv = np.array(result)
    resultcv = cv2.resize(resultcv, (640,640))

    return resultcv

def run(ros_goruntu):

    startStop_coord = []

    im = np.frombuffer(ros_goruntu.data, dtype=np.uint8).reshape(ros_goruntu.height, ros_goruntu.width, -1)
    RGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    combine = LaneSegmentation(RGB)
    '''
    finline = startStopSegmentation(RGB)

    x,y,w,h = mask_to_bbox(finline)
    x1 = x 
    y1 = y
    x2 = x1 + w
    y2 = y1 + h
                
    startStop_coord.append(int(x1))
    startStop_coord.append(int(y1))
    startStop_coord.append(int(x2))
    startStop_coord.append(int(y2))

    startStop_coordX = Int32MultiArray(data=startStop_coord)
    StartStop.publish(startStop_coordX)

    cv2.rectangle(combine, (x1,y1), (x2,y2), (255,0,0), 3)

    imgx = cv2_to_imgmsg(combine)
    imgx.header.stamp = rospy.Time.now()
    SegmentTalker.publish(imgx)
    
    '''
    '''
    cv2.imshow('frame', combine)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown('kapatiliyor...')
    '''

def main(args):
    rospy.init_node('talkerSeg', anonymous=True)
    rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, run)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatiliyor")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

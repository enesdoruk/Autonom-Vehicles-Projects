import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from tensorflow.python.keras.models import load_model
import time

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):


    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = imresize(lane_drawn, (416, 416, 3))

    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    cv2.imshow("image", result)

    return result


if __name__ == '__main__':
    model = load_model('full_CNN_model.h5')
    lanes = Lanes()

    btu = cv2.imread("btu.jpg")
    btu = cv2.resize(btu, dsize=(80,80))
    carf = cv2.imread("carf.jpg")
    carf = cv2.resize(carf, dsize=(80,80))
    
    vid = cv2.VideoCapture("project_video.mp4")

    while (True):
        ret, frame = vid.read()

        if ret == True:

            small_img = imresize(frame, (80, 160, 3))
            scaled = cv2.resize(frame,(416,416))
            small_img = np.array(small_img)
            small_img = small_img[None, :, :, :]

            prediction = model.predict(small_img)[0] * 255

            lanes.recent_fit.append(prediction)
            if len(lanes.recent_fit) > 5:
                lanes.recent_fit = lanes.recent_fit[1:]

            lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

            blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
            lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

            lane_image = imresize(lane_drawn, (416, 416, 3))

            result = cv2.addWeighted(scaled, 1, lane_image, 1, 0)
            
            mask = result.copy()
            mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(416, 100), color=(0, 0, 0), thickness=cv2.FILLED)
            blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=result, beta=0.8, gamma=0)
            blend_on_road[5:85, 10:90, :] = btu
            blend_on_road[5:85, 100:180, :] = carf

            cv2.imshow("image", blend_on_road)
            
            cv2.waitKey(0)

        #vid_output = 'proj_reg_vid.mp4'
    #clip1 = VideoFileClip("project_video.mp4")
    #vid_clip = clip1.fl_image(road_lines)
    #vid_clip.write_videofile(vid_output, audio=False)
import os
import numpy as np
import pickle
from util.functions_detection import *
import scipy
from util.functions_utils import normalize_image
from util.feature_extraction import find_cars
import time
import collections
from camera_calibration.calibration_process import calibrate_camera, undistort

time_window = 5
hot_windows_history = collections.deque(maxlen=time_window)

svc = pickle.load(open('data/svm_trained.pickle', 'rb'))

feature_scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))


def prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection):

    h, w, c = frame.shape

    thumb_ratio = 0.25
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    thumb_hot_windows = cv2.resize(img_hot_windows, dsize=(thumb_w, thumb_h))
    thumb_heatmap = cv2.resize(img_heatmap, dsize=(thumb_w, thumb_h))
    thumb_labeling = cv2.resize(img_labeling, dsize=(thumb_w, thumb_h))

    off_x, off_y = 20, 45

    mask = cv2.rectangle(img_detection.copy(), (0, 0), (2*off_x + thumb_w, h), (0, 0, 0), thickness=cv2.FILLED)
    img_blend = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_detection, beta=0.8, gamma=0)

    img_blend[off_y:off_y+thumb_h, off_x:off_x+thumb_w, :] = thumb_hot_windows
    img_blend[2*off_y+thumb_h:2*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_heatmap
    img_blend[3*off_y+2*thumb_h:3*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_labeling

    return img_blend


def process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False):

    hot_windows = []

    for subsample in np.arange(1, 3, 0.5):
        hot_windows += find_cars(frame, 400, 600, subsample, svc, feature_scaler, feat_extraction_params)

    if keep_state:
        if hot_windows:
            hot_windows_history.append(hot_windows)
            hot_windows = np.concatenate(hot_windows_history)

    thresh = (time_window - 1) if keep_state else 0
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=thresh, verbose=False)

    labeled_frame, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)

    img_hot_windows = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=2)                
    img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)       
    img_labeling = cv2.applyColorMap(normalize_image(labeled_frame), colormap=cv2.COLORMAP_HOT)  
    img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)       

    img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)

    if verbose:
        cv2.imshow('detection bboxes', img_hot_windows)
        cv2.imshow('heatmap', img_heatmap)
        cv2.imshow('labeled frame', img_labeling)
        cv2.imshow('detections', img_detection)
        cv2.waitKey()

    return img_blend_out


if __name__ == '__main__':

    cap = cv2.VideoCapture("data/test_video.mp4")
    prev_frame_time = 0
    new_frame_time = 0

    while(True):
        ret, frame = cap.read()
        t = time.time()
        new_frame_time = time.time()
        
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='../camera_cal')
        img_undist = undistort(frame, mtx, dist)
        frame_out = process_pipeline(img_undist, svc, feature_scaler, feat_extraction_params, keep_state=False, verbose=False)
        cv2.putText(frame_out, 'Detection boxes', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame_out, 'Heat Map', (20,260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame_out, 'Labeled Frame', (20,480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame_out, 'BTU Abdullah Enes DORUK', (900,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame_out, 'FPS: {}'.format(fps), (900, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Vehicle Detection', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print('Done. Elapsed: {:.02f}'.format(time.time()-t))
    
    cap.release()
    cv2.destroyAllWindows()


  


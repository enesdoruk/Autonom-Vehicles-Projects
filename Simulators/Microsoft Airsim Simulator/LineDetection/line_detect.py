import cv2
import os
import matplotlib.pyplot as plt
from LineDetection.calibration_process import calibrate_camera, undistort
from LineDetection.binarization_process import binarize
from LineDetection.perspective_process import birdeye
from LineDetection.line_process import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits
import numpy as np
import time


ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700
time_window = 10

processed_frames = 0
line_lt = Line(buffer_len=time_window)
line_rt = Line(buffer_len=time_window)


def prepare_out_blend_frame(blend_on_road, img_fit, line_lt, line_rt, offset_meter):

    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Egri Yaricap: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Merkezden Ofset: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def compute_offset_from_center(line_lt, line_rt, frame_width):

    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames

    img_binary = binarize(frame, verbose=False)

    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit, x, y = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    blend_on_road = draw_back_onto_the_road(frame, Minv, line_lt, line_rt, keep_state, x, y)

    blend_output = prepare_out_blend_frame(blend_on_road, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    return blend_output ,x, y


import os
import sys
from os.path import exists
from os.path import join

import cv2
import numpy as np


def get_file_list_recursively(top_directory):
    
    if not exists(top_directory):
        raise ValueError('Directory "{}" does NOT exist.'.format(top_directory))

    file_list = []

    for cur_dir, cur_subdirs, cur_files in os.walk(top_directory):

        for file in cur_files:
            file_list.append(join(cur_dir, file))
            sys.stdout.write(
                '\r[{}] - found {:06d} files...'.format(top_directory, len(file_list)))
            sys.stdout.flush()

    sys.stdout.write(' Done.\n')

    return file_list


def stitch_together(input_images, layout, resize_dim=None, off_x=None, off_y=None,
                    bg_color=(0, 0, 0)):
    

    if len(set([img.shape for img in input_images])) > 1:
        raise ValueError('All images must have the same shape')

    if len(set([img.dtype for img in input_images])) > 1:
        raise ValueError('All images must have the same data type')

    if len(input_images[0].shape) == 2:
        mode = 'grayscale'
        img_h, img_w = input_images[0].shape
    elif len(input_images[0].shape) == 3:
        mode = 'color'
        img_h, img_w, img_c = input_images[0].shape
    else:
        raise ValueError('Unknown shape for input images')

    if off_x is None:
        off_x = img_w // 10
    if off_y is None:
        off_y = img_h // 10

    rows, cols = layout
    stitch_h = rows * img_h + (rows + 1) * off_y
    stitch_w = cols * img_w + (cols + 1) * off_x
    if mode == 'color':
        bg_color = np.array(bg_color)[None, None, :]  
        stitch = np.uint8(np.repeat(np.repeat(bg_color, stitch_h, axis=0), stitch_w, axis=1))
    elif mode == 'grayscale':
        stitch = np.zeros(shape=(stitch_h, stitch_w), dtype=np.uint8)

    for r in range(0, rows):
        for c in range(0, cols):

            list_idx = r * cols + c

            if list_idx < len(input_images):
                if mode == 'color':
                    stitch[r * (off_y + img_h) + off_y: r * (off_y + img_h) + off_y + img_h,
                    c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w,
                    :] = input_images[list_idx]
                elif mode == 'grayscale':
                    stitch[r * (off_y + img_h) + off_y: r * (off_y + img_h) + off_y + img_h,
                    c * (off_x + img_w) + off_x: c * (off_x + img_w) + off_x + img_w] \
                        = input_images[list_idx]

    if resize_dim:
        stitch = cv2.resize(stitch, dsize=(resize_dim[::-1]))

    return stitch


class Rectangle:
   
    def __init__(self, x_min, y_min, x_max, y_max, label=""):

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.x_side = self.x_max - self.x_min
        self.y_side = self.y_max - self.y_min

        self.label = label

    def intersect_with(self, rect):
      
        if not isinstance(rect, Rectangle):
            raise ValueError('Cannot compute intersection if "rect" is not a Rectangle')

        dx = min(self.x_max, rect.x_max) - max(self.x_min, rect.x_min)
        dy = min(self.y_max, rect.y_max) - max(self.y_min, rect.y_min)

        if dx >= 0 and dy >= 0:
            intersection = dx * dy
        else:
            intersection = 0.

        return intersection

    def resize_sides(self, ratio, bounds=None):
       
        off_x = abs(ratio * self.x_side - self.x_side) / 2
        off_y = abs(ratio * self.y_side - self.y_side) / 2

        sign = np.sign(ratio - 1.)
        off_x = np.int32(off_x * sign)
        off_y = np.int32(off_y * sign)

        new_x_min, new_y_min = self.x_min - off_x, self.y_min - off_y
        new_x_max, new_y_max = self.x_max + off_x, self.y_max + off_y

        if bounds:
            b_x_min, b_y_min, b_x_max, b_y_max = bounds
            new_x_min = max(new_x_min, b_x_min)
            new_y_min = max(new_y_min, b_y_min)
            new_x_max = min(new_x_max, b_x_max)
            new_y_max = min(new_y_max, b_y_max)

        return Rectangle(new_x_min, new_y_min, new_x_max, new_y_max)

    def draw(self, frame, color=255, thickness=2, draw_label=False):
       
        if draw_label and self.label:
            text_font, text_scale, text_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (text_w, text_h), baseline = cv2.getTextSize(self.label, text_font, text_scale,
                                                         text_thick)

            text_rect_w = min(text_w, self.x_side - 2 * baseline)
            out = cv2.rectangle(frame.copy(), pt1=(self.x_min, self.y_min - text_h - 2 * baseline),
                                pt2=(self.x_min + text_rect_w + 2 * baseline, self.y_min),
                                color=color, thickness=cv2.FILLED)
            cv2.addWeighted(frame, 0.75, out, 0.25, 0, dst=frame)

            cv2.putText(frame, self.label, (self.x_min + baseline, self.y_min - baseline),
                        text_font, text_scale, (0, 0, 0), text_thick, cv2.LINE_AA)

            cv2.rectangle(frame, pt1=(self.x_min, self.y_min - text_h - 2 * baseline),
                          pt2=(self.x_min + text_rect_w + 2 * baseline, self.y_min), color=color,
                          thickness=thickness)

        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max), color, thickness)

    def get_binary_mask(self, mask_shape):
      
        if mask_shape[0] < self.y_max or mask_shape[1] < self.x_max:
            raise ValueError('Mask shape is smaller than Rectangle size')
        mask = np.zeros(shape=mask_shape, dtype=np.uint8)
        mask = cv2.rectangle(mask, self.tl_corner, self.br_corner, color=255, thickness=cv2.FILLED)
        return mask

    @property
    def tl_corner(self):
       
        return tuple(map(np.int32, (self.x_min, self.y_min)))

    @property
    def br_corner(self):
        
        return tuple(map(np.int32, (self.x_max, self.y_max)))

    @property
    def coords(self):
       
        return tuple(map(np.int32, (self.x_min, self.y_min, self.x_max, self.y_max)))

    @property
    def area(self):
      
        return np.float32(self.x_side * self.y_side)

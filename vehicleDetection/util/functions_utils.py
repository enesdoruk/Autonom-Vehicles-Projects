import numpy as np

def normalize_image(img):
   
    img = np.float32(img)
    img = img / img.max() * 255

    return np.uint8(img)
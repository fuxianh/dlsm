import numpy as np
from crop import im_crop


def get_batch(images, boxes, opts):
    im_list = []
    for image, bbox in zip(images, boxes):
        im = im_crop(image, bbox, opts.crop_mode, opts.crop_size, opts.padding, opts.mean_rgb)
        im_list.append(im)
    return im_list

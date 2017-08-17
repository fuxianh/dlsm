import cv2
import numpy as np


def im_crop(im, bbox, crop_mode, crop_size, padding, mean_rgb=None):
    """ Crops a window specified by bbox (in [x, y, w, h] order) out of im.
    crop_mode: can be either 'warp' or 'square'
    crop_size: determines the size of the output window: crop_size x crop_size
    padding: is the amount of padding to include at the target scale
    mean_rgb: to subtract from the cropped window
    """
    im_h, im_w, im_c = im.shape
    use_square = True if crop_mode == 'square' else False

    bbox_ = bbox.copy()
    pad_w, pad_h = 0, 0
    crop_w, crop_h = crop_size, crop_size

    if padding > 0 or use_square:
        scale = crop_size / (crop_size - padding * 2.)
        half_w = bbox[2] / 2.
        half_h = bbox[3] / 2.
        center = np.array([bbox[0] + half_w, bbox[1] + half_h])

        if use_square:
            if half_h > half_w:
                half_w = half_h
            else:
                half_h = half_w

        bbox_ = np.round(np.append(center, center) +
                         np.array([-half_w, -half_h, half_w, half_h]) * scale).astype(np.int)
        unclipped_w = bbox_[2] - bbox_[0] + 1
        unclipped_h = bbox_[3] - bbox_[1] + 1
        pad_x1 = max(0, 1 - bbox_[0])
        pad_y1 = max(0, 1 - bbox_[1])
        # clipped bbox
        bbox_[0] = max(0, bbox_[0])
        bbox_[1] = max(0, bbox_[1])
        bbox_[2] = min(im_w, bbox_[2])
        bbox_[3] = min(im_h, bbox_[3])
        clipped_h = bbox_[3] - bbox_[1] + 1
        clipped_w = bbox_[2] - bbox_[0] + 1
        scale_x = float(crop_size) / unclipped_w
        scale_y = float(crop_size) / unclipped_h
        crop_w = np.round(clipped_w * scale_x)
        crop_h = np.round(clipped_h * scale_y)
        pad_x1 = np.round(pad_x1 * scale_x)
        pad_y1 = np.round(pad_y1 * scale_y)

        pad_h = int(pad_y1)
        pad_w = int(pad_x1)

        if pad_y1 + crop_h > crop_size:
            crop_h = crop_size - pad_y1
        if pad_x1 + crop_w > crop_size:
            crop_w = crop_size - pad_x1
            
    crop_w = int(crop_w)
    crop_h = int(crop_h)
    window = im[bbox_[1]: bbox_[3], bbox_[0]: bbox_[2], :]
    tmp = cv2.resize(window, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    tmp -= 128 if mean_rgb is None else mean_rgb

    window = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    window[pad_h: pad_h+crop_h, pad_w: pad_w + crop_w, :] = tmp
    return window

if __name__ == '__main__':
    from vot import load_vot
    import matplotlib.pyplot as plt
    vdb = load_vot('D:\\dataset\\vot2014')
    vid = vdb['ball']
    entry = vid[0]

    im = cv2.imread(entry['im_path'])
    bbox= entry['bbox']

    window = im_crop(im, bbox, 'warp', 107, 16)
    plt.imshow(window[:,:,(2,1,0)])
    plt.show()
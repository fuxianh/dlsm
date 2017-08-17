import os.path as osp
from PIL import Image
import numpy as np
from numpy import loadtxt as npl
import StringIO


def load_video(dataset, vdir):
    """Load single video with images and ground-truth."""
    # Step 1: load ground-truth and images
    s = open(osp.join(dataset, vdir, 'groundtruth.txt')).read().replace(',', ' ')
    bboxes = npl(StringIO.StringIO(s), dtype=np.float32)
    num = bboxes.shape[0]
    vimgs = [osp.join(dataset, vdir, '{:08d}.jpg'.format(i)) for i in range(1, num + 1)]
    im_size = Image.open(vimgs[0]).size

    x = bboxes[:, (0, 2, 4, 6)]
    x[x < 1] = 1
    x[x > im_size[0]] = im_size[0]
    y = bboxes[:, (1, 3, 5, 7)]
    y[y < 1] = 1
    y[y > im_size[1]] = im_size[1]

    x1 = np.min(x, axis=1)[:, np.newaxis]
    x2 = np.max(x, axis=1)[:, np.newaxis]
    y1 = np.min(y, axis=1)[:, np.newaxis]
    y2 = np.max(y, axis=1)[:, np.newaxis]
    w = x2 - x1
    h = y2 - y1
    bboxes = np.hstack((x1, y1, w, h))

    entries = []
    for bbox, im_path in zip(bboxes, vimgs):
        entries.append({
            'bbox': bbox,
            'im_path': im_path,
            'im_size': im_size
        })

    return entries


def load_vot(dataset):
    """Load VOT video dataset"""
    vdb = {}
    # Read video list
    with open(osp.join(dataset, 'list.txt')) as f:
        vdirs = [vdir.strip() for vdir in f.readlines()]
    for vdir in vdirs:
        vdb[vdir] = load_video(dataset, vdir)
    return vdb


def _vis_data(vdb, vname):
    import matplotlib.pyplot as plt
    vid = vdb[vname]

    for entry in vid:
        im_path = entry['im_path']
        gt = entry['bboxes'][0]
        plt.cla()
        im = plt.imread(im_path)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.gca().add_patch(plt.Rectangle(
            (gt[0], gt[1]), gt[2], gt[3],
            linewidth=1.5, color='red', fill=False
        ))
        plt.pause(.1)
        plt.draw()

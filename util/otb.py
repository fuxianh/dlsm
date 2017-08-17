import os
import os.path as osp
from PIL import Image
import numpy as np
from numpy import loadtxt as npl
import StringIO

SPECIAL_VIDS = {
    'David': {
        'begin': 300,
        'end': 770
    },
    'Tiger1':{
        'begin': 6,
        'end': 355
    },
    'Football1':{
        'begin': 1,
        'end': 74
    },
    'Freeman3':{
        'begin': 1,
        'end': 460
    },
    'Freeman4':{
        'begin': 1,
        'end': 283
    }
}


def load_video(dataset, vdir):
    """Load single video with images and ground-truth."""
    # Step 1: load ground-truth
    if vdir == 'Jogging-1' or vdir == 'Skating2-1':
        vdir = vdir[:-2]
        s = open(osp.join(dataset, vdir, 'groundtruth_rect.1.txt')).read().replace(',', ' ')
    elif vdir == 'Jogging-2' or vdir == 'Skating2-2':
        vdir = vdir[:-2]
        s = open(osp.join(dataset, vdir, 'groundtruth_rect.2.txt')).read().replace(',', ' ')
    else:
        s = open(osp.join(dataset, vdir, 'groundtruth_rect.txt')).read().replace(',', ' ')

    bboxes = npl(StringIO.StringIO(s),
                 dtype=np.float32)

    # Step 2: load images
    vimg_dir = osp.join(dataset, vdir, 'img')
    img_list = os.listdir(vimg_dir)
    # Sort images to get the correct order
    img_list.sort()
    vimgs = [osp.join(vimg_dir, img) for img in img_list]
    if vdir in SPECIAL_VIDS:
        vid = SPECIAL_VIDS[vdir]
        vimgs = vimgs[vid['begin']-1: vid['end']]
        if vdir == 'Tiger1':
            bboxes = bboxes[vid['begin']-1: vid['end']]
    vimgs = vimgs[: len(bboxes)]

    assert len(bboxes) == len(vimgs), \
        "[Load OTB] {}: Number of ground-truth and that " \
        "of images don't match with {} vs {}".format(vdir, len(bboxes), len(vimgs))

    entries = []
    # Take the size of the first frame as the size of all the frames
    im_size = Image.open(vimgs[0]).size
    # Transform 1-base to 0-base
    bboxes[:, 0] -= 1
    bboxes[:, 1] -= 1
    for bbox, im_path in zip(bboxes, vimgs):
        entries.append({
            'bbox': bbox,
            'im_path': im_path,
            'im_size': im_size
        })

    return entries


def load_otb50(dataset):
    """Load OTB 50 video dataset"""
    vdb = {}
    # list video directories of the database
    with open(osp.join(dataset, 'seq_list_50.txt')) as f:
        vdirs = [vdir.strip() for vdir in f.readlines()]
    for vdir in vdirs:
        vdb[vdir] = load_video(dataset, vdir)
    return vdb


def load_otb(dataset):
    vdb = {}
    # list video directories of the database
    dirs = os.listdir(dataset)
    vdirs = []
    for _dir in dirs:
        if _dir.endswith('txt'):
            continue
        if _dir == 'Jogging':
            vdirs.append('Jogging-1')
            vdirs.append('Jogging-2')
        elif _dir == 'Skating2':
            vdirs.append('Skating2-1')
            vdirs.append('Skating2-2')
        else:
            vdirs.append(_dir)
    for vdir in vdirs:
        vdb[vdir] = load_video(dataset, vdir)
    return vdb


def _vis_data(vdb, vname):
    import matplotlib.pyplot as plt
    vid = vdb[vname]
    gt_rects = vid['gt_rects']
    vimgs = vid['imgs']

    for gt, f in zip(gt_rects, vimgs):
        plt.cla()
        im = plt.imread(f)
        plt.imshow(im)
        plt.gca().add_patch(plt.Rectangle(
            (gt[0], gt[1]), gt[2], gt[3],
            linewidth=1.5, color='red', fill=False
        ))
        plt.pause(.1)
        plt.draw()

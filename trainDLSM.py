
# coding: utf-8

# # Demo for training DLSM

# In[1]:

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import os, sys
import numpy as np
import cv2

from config import opts
from util.vot import load_vot
from util.batch import get_batch
from util.blob import im_list_to_blob

caffe.set_mode_gpu()
caffe.set_device(0)


# In[2]:

DATABASE = 'vot2016'
CNN_NET = 'MDNet/deploy_test.prototxt'
CNN_WEIGHTS = 'out/vot2016_final.caffemodel'

cnn_net = caffe.Net(os.path.join('prototxts', CNN_NET), CNN_WEIGHTS, caffe.TEST)

PRED_SOLVER = 'DLSM/pred_solver.prototxt'
pred_solver = caffe.SGDSolver(os.path.join('prototxts', PRED_SOLVER))

EVL_SOLVER = 'DLSM/evl_solver.prototxt'
evl_solver = caffe.SGDSolver(os.path.join('prototxts', EVL_SOLVER))


# In[3]:

# Load video database
vdb = load_vot(os.path.join('data', DATABASE))
with open(opts.exclude) as f:
    exclude = [l.strip() for l in f.readlines() if len(l.strip())]
    for exc in exclude:
        if exc in vdb: del vdb[exc]


# In[4]:

def transformBoxes(bboxes, im_w, im_h):
    bboxes[:, 0] += bboxes[:, 2] / 2.
    bboxes[:, 1] += bboxes[:, 3] / 2.
    bboxes[:, 0] /= im_w
    bboxes[:, 1] /= im_h
    bboxes[:, 2] /= im_w
    bboxes[:, 3] /= im_h

    return bboxes


def entries2batch(entries):
    train_entries, label_entries = entries[:-1], entries[1:]
    images = []
    # build delta blob
    bboxes = np.zeros((0, 4), dtype=np.float32)
    for ix, entry in enumerate(train_entries):
        if ix>0: images.append(cv2.imread(entry['im_path']))
        bboxes = np.vstack((bboxes, entry['bbox']))
    im_h, im_w, _ = images[0].shape
    boxes = bboxes[:-1].copy()
    bboxes = transformBoxes(bboxes, im_w, im_h)
    delta_blob = np.diff(bboxes, axis=0)
    # build label blob 
    bboxes = np.zeros((0, 4), dtype=np.float32)
    for ix, entry in enumerate(label_entries):
        bboxes = np.vstack((bboxes, entry['bbox']))
    bboxes = transformBoxes(bboxes, im_w, im_h)
    label_blob = np.diff(bboxes, axis=0)
    # build data blob for CNN
    im_list = get_batch(images, boxes, opts)
    im_blob = im_list_to_blob(im_list)
    # build clip blob
    clip_blob = np.ones((delta_blob.shape[0], 1), dtype=np.float32)
    clip_blob[0, 0] = 0
    
    return {'data': im_blob}, {'delta_pred': delta_blob.reshape(-1,1,4), 
                                'label_pred': label_blob.reshape(-1,4),
                                'clip_pred': clip_blob}


def main():
    for c in range(opts.dlsm_cycle):
        for vkey, entries in vdb.items():
            im_h, im_w, _ = cv2.imread(entries[0]['im_path']).shape
            for i in range(len(entries) - opts.seq_len - 1):
                # build blobs for prediction LSTM
                im_blob, pred_blob = entries2batch(entries[i: i + opts.seq_len + 1])
                for blob_name, blob_data in im_blob.items():
                    cnn_net.blobs[blob_name].reshape(*blob_data.shape)
                    cnn_net.blobs[blob_name].data[...] = blob_data
                _ = cnn_net.forward()
                conv3_blob = cnn_net.blobs['conv3'].data
                pred_blob['conv3'] = conv3_blob.reshape(1, opts.seq_len - 1, -1)

                for blob_name, blob_data in pred_blob.items():
                    pred_solver.net.blobs[blob_name].reshape(*blob_data.shape)
                    pred_solver.net.blobs[blob_name].data[...] = blob_data
                pred_solver.step(1)

                # build blobs for refinement LSTM
                delta_err = np.abs(pred_solver.net.blobs['ip_pred'].data - pred_blob['label_pred'])
                delta_err[:, 0] += opts.ref_pad / im_w
                delta_err[:, 1] += opts.ref_pad / im_h
                delta_err[:, 2] += opts.ref_pad / im_w
                delta_err[:, 3] += opts.ref_pad / im_h
                delta_blob = delta_err[:-1].reshape(-1, 1, 4)
                label_blob = delta_err[1:].reshape(-1, 4)
                clip_blob = np.ones((delta_blob.shape[0], 1), dtype=np.float32)
                clip_blob[0, 0] = 0
                evl_blobs = {'delta_evl': delta_blob, 'label_evl': label_blob, 'clip_evl': clip_blob}
                for blob_name, blob_data in evl_blobs.items():
                    evl_solver.net.blobs[blob_name].reshape(*blob_data.shape)
                    evl_solver.net.blobs[blob_name].data[...] = blob_data
                evl_solver.step(1)

        pred_solver.net.save(str('out/pred_{}.caffemodel'.format(c + 1)))
        evl_solver.net.save(str('out/evl_{}.caffemodel'.format(c + 1)))

if __name__ == '__main__':
    main()

import os, sys
import cv2
import numpy as np
import pickle as pkl

from config import opts
from util.seq2roidb import gen_roidb
from util.wrappers import TimeWrapper
from util.batch import get_batch
from util.blob import im_list_to_blob

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2

caffe.set_mode_gpu()
caffe.set_device(0)


# In[2]:

def load_roidb(database):
    if 'OTB50' in database.upper():
        dtype = 'OTB50'
    elif 'OTB' in database.upper():
        dtype = 'OTB'
    elif 'VOT' in database.upper():
        dtype = 'VOT'
    else:
        assert False, 'Unknown database {}'.format(database)

    timer = TimeWrapper()
    timer.tic()
    cache_file = os.path.join('data', '{}.pkl'.format(database))
    if os.path.exists(cache_file):
        print 'Cache file {} exists and we load roidb from the cache file.'.format(cache_file)
        with open(cache_file) as f:
            roidb = pkl.load(f)
    else:
        roidb = gen_roidb(os.path.join('data', database), opts, dtype)
        # construct training roidb
        train_roidb = list()
        for ventry in roidb:
            this_roidb = list()
            n_frames = opts.train.batch_frames * opts.train.num_cycles
            while len(this_roidb) < n_frames:
                this_roidb.extend(np.random.permutation(ventry['entries']))
            this_roidb = this_roidb[: n_frames]
            train_roidb.append(this_roidb)
        roidb = train_roidb
        # store roidb to a cache file
        with open(cache_file, 'wb') as f:
            pkl.dump(obj=roidb, file=f)
    timer.toc()

    print 'Time for loading roidb: {}'.format(timer.total_time)

    return roidb


# In[3]:

DATABASE = 'OTB50'
SOLVER = 'MDNet/presolver.prototxt'
WEIGHTS = 'VGG_CNN_M_1024.caffemodel'

roidb = load_roidb(DATABASE)
out_dir = 'out'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


# In[4]:

solver_prototxt = os.path.join('prototxts', SOLVER)
weights = os.path.join('models', WEIGHTS)
solver = caffe.SGDSolver(solver_prototxt)
if weights is not None:
    solver.net.copy_from(weights)


# In[5]:

def entries2batch(entries):
    im_list = [cv2.imread(entry['im_path']) for entry in entries]
    pos_boxes = np.zeros((0, 5), dtype=np.float32) # [im_idx x y w h]
    neg_boxes = np.zeros((0, 5), dtype=np.float32) # [im_idx x y w h]
    
    # build pos_boxes and neg_boxes
    for idx, entry in enumerate(entries):
        this_pos_boxes = entry['pos_boxes']
        pos_im_idx = np.ones((this_pos_boxes.shape[0], 1), dtype=np.float32) * idx
        this_pos_boxes = np.hstack((pos_im_idx, this_pos_boxes))
        pos_boxes = np.vstack((pos_boxes, this_pos_boxes))
        this_neg_boxes = entry['neg_boxes']
        neg_im_idx = np.ones((this_neg_boxes.shape[0], 1), dtype=np.float32) * idx
        this_neg_boxes = np.hstack((neg_im_idx, this_neg_boxes))
        neg_boxes = np.vstack((neg_boxes, this_neg_boxes))
    
    # random permutate the boxes and choose some of them as the batch for training
    pos_samples = np.random.permutation(pos_boxes)[: opts.train.batch_pos]
    neg_samples = np.random.permutation(neg_boxes)[: opts.train.batch_neg]
    
    pos_ims = list()
    pos_boxes = list()
    for pos_sample in pos_samples:
        pos_ims.append(im_list[int(pos_sample[0])])
        pos_boxes.append(pos_sample[1:])
        
    neg_ims = list()
    neg_boxes = list()
    for neg_sample in neg_samples:
        neg_ims.append(im_list[int(neg_sample[0])])
        neg_boxes.append(neg_sample[1:])
        
    ims = pos_ims + neg_ims
    boxes = pos_boxes + neg_boxes
    # build image blob and label blob
    im_list = get_batch(ims, boxes, opts)
    im_blob = im_list_to_blob(im_list)
    labels = np.hstack((np.ones(len(pos_boxes), dtype=np.float32), 
                       np.zeros(len(neg_boxes), dtype=np.float32)))
    blobs = {
        'data': im_blob,
        'labels': labels
    }
    
    return blobs


# In[6]:

params = []
for i in range(len(roidb)):
    params.append({
            'params': (0.01 * np.random.randn(2, 512)).astype(dtype=np.float32),
            'consts': np.zeros(2, dtype=np.float32)
        })


# ## Traing code, and we only use one sequence for training

# In[7]:

for t in range(opts.train.num_cycles):
    print 'Training: processing cycle {} of {}'.format(t+1, opts.train.num_cycles)
    for seqID in range(len(roidb)):
        print 'Training: processing training entries {} of {}'.format(seqID+1, len(roidb))
        train_entries = roidb[seqID]
        solver.net.params['cls_score'][0].data[...] = params[seqID]['params']
        solver.net.params['cls_score'][1].data[...] = params[seqID]['consts']
        
        cur = 0
        while cur + opts.train.batch_frames <= len(train_entries):
            entries = train_entries[cur: cur + opts.train.batch_frames]
            blobs = entries2batch(entries)
            for blob_name, blob_data in blobs.items():
                solver.net.blobs[blob_name].reshape(*blob_data.shape)
                solver.net.blobs[blob_name].data[...] = blob_data
            solver.step(1)
            cur += opts.train.batch_frames
        
        params[seqID]['params'] = solver.net.params['cls_score'][0].data.copy()
        params[seqID]['consts'] = solver.net.params['cls_score'][1].data.copy()
    if (t+1) % 10 == 0:
    	solver.net.save(str('{}_{}.caffemodel'.format(DATABASE, t+1)))



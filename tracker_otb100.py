
# coding: utf-8

# In[1]:

import os, sys
import cv2
import numpy as np
import pickle as pkl

from config import opts
from util.otb import load_otb
from util.seq2roidb import gen_roidb
from util.wrappers import TimeWrapper
from util.batch import get_batch
from util.blob import im_list_to_blob
from util.bbox import gen_samples
from util.bbox import overlap_ratio

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2

caffe.set_mode_gpu()
caffe.set_device(1)


# In[2]:

DATABASE = 'data/OTB50'
vdb = load_otb(DATABASE)


# In[ ]:

SOLVER = 'MDNet/deploy_solver.prototxt'
WEIGHTS = 'vot2016_final.caffemodel'
NET = 'MDNet/deploy_test.prototxt'

def initSolverNet():
    solver_prototxt = os.path.join('prototxts', SOLVER)
    net_prototxt = os.path.join('prototxts', NET)
    weights = os.path.join('out', WEIGHTS)
    solver = caffe.SGDSolver(solver_prototxt)
    solver.net.copy_from(weights)
    net = caffe.Net(net_prototxt, caffe.TEST)
    net.share_with(solver.net)

    solver.net.params['cls_score'][0].data[...] = (0.01 * np.random.randn(2, 512)).astype(dtype=np.float32)
    solver.net.params['cls_score'][1].data[...] = np.zeros(2, dtype=np.float32)
    
    return solver, net


# In[ ]:

def initFinetune(solver, im_init, bb_init):
    im = cv2.imread(im_init)
    # draw positive and negative samples
    pos_samples = gen_samples(im, 'gaussian', bb_init, opts.nPos_init*2 ,opts ,0.1, 5)
    ratios = np.zeros((0), dtype=np.float32)
    for _pos_bb in pos_samples:
        ratios = np.hstack((ratios, overlap_ratio(_pos_bb, bb_init)))
    pos_samples = pos_samples[ratios>opts.posThr_init]
    pos_samples = pos_samples[np.random.permutation(np.arange(len(pos_samples)))[: min(opts.nPos_init, len(pos_samples))]]

    neg_samples = np.vstack((gen_samples(im, 'uniform', bb_init, opts.nNeg_init, opts, 1, 10),
                            gen_samples(im, 'whole', bb_init, opts.nNeg_init, opts, None, None)))
    ratios = np.zeros((0), dtype=np.float32)
    for _neg_bb in neg_samples:
        ratios = np.hstack((ratios, overlap_ratio(_neg_bb, bb_init)))
    neg_samples = neg_samples[ratios<opts.negThr_init]
    neg_samples = neg_samples[np.random.permutation(np.arange(len(neg_samples)))[: min(opts.nNeg_init, len(neg_samples))]]
    
    # build init entry and finetune the net
    init_entry = [{
        'im_path': im_init,
        'pos_boxes': pos_samples,
        'neg_boxes': neg_samples
    }]
    for i in range(30):
        blobs = entries2batch(init_entry)
        for blob_name, blob_data in blobs.items():
            solver.net.blobs[blob_name].reshape(*blob_data.shape)
            solver.net.blobs[blob_name].data[...] = blob_data
        solver.step(1)
    return pos_samples, neg_samples


# In[ ]:

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


# In[ ]:

for vname, trackEntries in vdb.items():
    print 'Begin Tracking video {}'.format(vname)
    results = np.zeros((0, 4), dtype=np.float32)
    
    solver, net = initSolverNet()
    im_init = trackEntries[0]['im_path']
    bb_init = trackEntries[0]['bbox']
    pos_samples, neg_samples = initFinetune(solver, im_init, bb_init)

    trans_f, scale_f = opts.trans_f, opts.scale_f
    targetLoc = bb_init
    results = np.vstack((results, targetLoc))
    # prepare training data for online update
    total_data = [{'im_path': im_init, 
                   'pos_boxes': pos_samples[np.random.permutation(np.arange(len(pos_samples)))[: min(opts.nPos_update, len(pos_samples))]], 
                   'neg_boxes': neg_samples[np.random.permutation(np.arange(len(neg_samples)))[: min(opts.nNeg_update, len(neg_samples))]]}]
    
    for frameID, entry in enumerate(trackEntries[1:]):
        im = cv2.imread(entry['im_path'])
        samples = gen_samples(im, 'gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f)
    
        im_list = get_batch([im]*len(samples), samples, opts)
        im_blob = im_list_to_blob(im_list)
        net.blobs['data'].reshape(*im_blob.shape)
        net.blobs['data'].data[...] = im_blob
        out = net.forward()
        scores = out['cls_prob'][:,1]
        sortIdxs = np.argsort(scores)[::-1]

        targetScore = np.mean(scores[sortIdxs[:5]])
        targetLoc = np.mean(samples[sortIdxs[:5]], axis=0)
        results = np.vstack((results, targetLoc))
    
        trans_f = min(1.5, 1.1 * trans_f) if targetScore <= 0.5 else opts.trans_f
    
        # prepare training data
        if targetScore > 0.5:
            pos_samples = gen_samples(im, 'gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5)
            ratios = np.array([overlap_ratio(_pos, targetLoc) for _pos in pos_samples])
            pos_samples = pos_samples[ratios > opts.posThr_update]
            pos_samples = pos_samples[np.random.permutation(np.arange(len(pos_samples)))[: min(opts.nPos_update, len(pos_samples))]]
        
            neg_samples = gen_samples(im, 'uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5)
            ratios = np.array([overlap_ratio(_neg, targetLoc) for _neg in neg_samples])
            neg_samples = neg_samples[ratios < opts.negThr_update]
            neg_samples = neg_samples[np.random.permutation(np.arange(len(neg_samples)))[: min(opts.nNeg_update, len(neg_samples))]]
        
            total_data.append({'im_path': entry['im_path'], 'pos_boxes': pos_samples, 'neg_boxes': neg_samples})
        else:
            total_data.append(None)
    
        # network update
        if ((frameID + 2) % opts.update_interval == 0 or targetScore <= 0.5) and frameID+1!=len(trackEntries):
            if targetScore <= 0.5:
                thisEntries = [_entry for _entry in total_data[-opts.nFrames_short:] if _entry]
            else:
                thisEntries = [_entry for _entry in total_data[-opts.nFrames_long:] if _entry]
            for i in range(20):
                blobs = entries2batch(thisEntries)
                for blob_name, blob_data in blobs.items():
                    solver.net.blobs[blob_name].reshape(*blob_data.shape)
                    solver.net.blobs[blob_name].data[...] = blob_data
                solver.step(1)
    
    with open('results/{}.pkl'.format(vname)) as f:
        pkl.dump(obj=results, file=f)


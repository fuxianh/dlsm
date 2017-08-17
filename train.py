import os, sys
import argparse
import pprint
import numpy as np
import pickle as pkl

from config import opts
from util.seq2roidb import gen_roidb
from util.wrappers import TimeWrapper

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--database', dest='database',
                        help='database to train on',
                        default='OTB50', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def load_roidb(database):
    if 'OTB' in database.upper():
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


def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    roidb = load_roidb(args.database)

    out_dir = 'out'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    solver_prototxt = os.path.join('prototxts', args.solver)
    weights = None if args.weights is None else os.path.join('models', args.weights)
    solver = caffe.SGDSolver(solver_prototxt)
    if weights is not None:
        solver.net.copy_from(weights)

if __name__ == '__main__':
    main()

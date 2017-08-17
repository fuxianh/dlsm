from easydict import EasyDict


opts = EasyDict()

opts.output_dir = 'out'  # directory for output
opts.data_dir = 'data'  # directory for dataset

opts.crop_mode = 'warp'
opts.crop_size = 107
opts.padding = 16
opts.mean_rgb = None 

opts.train = EasyDict()
opts.train.batch_frames = 8  # the number of frames to construct a minibatch
opts.train.batch_size = 128
opts.train.batch_pos = 32
opts.train.batch_neg = 96
opts.train.num_cycles = 100  # cycles (#iterations / #domains)
opts.train.use_GPU = True
opts.exclude = 'vot2016_exclude.txt'

opts.sampling = EasyDict()
opts.sampling.pos_perFrame = 50
opts.sampling.neg_perFrame = 200
opts.sampling.scale_factor = 1.05
opts.sampling.pos_range = [0.7, 1]
opts.sampling.neg_range = [0, 0.5]

opts.nPos_init = 500
opts.nNeg_init = 5000
opts.posThr_init = 0.7
opts.negThr_init = 0.5

# update policy
opts.learningRate_update = 0.0003 # x10 for fc6
opts.maxiter_update = 10

opts.nPos_update = 50
opts.nNeg_update = 200
opts.posThr_update = 0.7
opts.negThr_update = 0.3

opts.update_interval = 10 # interval for long-term update

# data gathering policy
opts.nFrames_long = 100 # long-term period
opts.nFrames_short = 20 # short-term period

# sampling policy
opts.nSamples = 256
opts.trans_f = 0.6 # translation std: mean(width,height)*trans_f/2
opts.scale_f = 1 # scaling std: scale_factor^(scale_f/2)

opts.dlsm_cycle = 10
opts.seq_len = 5 # sequence length for each batch leanring
opts.ref_pad = 10 # padding for refinement LSTM
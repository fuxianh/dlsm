import cv2
import numpy as np

from otb import load_otb50, load_otb
from vot import load_vot
from bbox import overlap_ratio


def gen_roidb(data_dir, opts, dtype):
    """Generate region-of-interest database. Return roidb in form:
    [ventry1, ventry2, ...] where ventry is in form:
       {
        'name': video name,
        'entries': [{'im_path': image path, 'pos_boxes': positive boxes, 'neg_boxes': negative boxes},
                    ...]
        }
    """
    if dtype == 'OTB':
        vdb = load_otb(data_dir)
    elif dtype == 'OTB50':
        vdb = load_otb50(data_dir)
    elif dtype == 'VOT':
        vdb = load_vot(data_dir)
    else:
        assert False, 'Unknown dataset {}'.format(dtype)

    exclude = []
    if opts.has_key('exclude'):
        with open(opts.exclude) as f:
            exclude = [l.strip() for l in f.readlines() if len(l.strip())]

    roidb = list()
    for vkey, vid in vdb.iteritems():
        if vkey in exclude:
            continue
        ventry = dict()
        ventry['name'] = vkey
        ventry['entries'] = list()
        im_shape = cv2.imread(vid[0]['im_path']).shape
        print 'Begin generate roidb for {}'.format(vkey)
        for ix, entry in enumerate(vid):
            bbox = entry['bbox']
            if bbox[2] < 10 or bbox[3] < 10: continue

            pos_examples = np.zeros((0, 4), dtype=np.float32)
            while pos_examples.shape[0] < opts.sampling.pos_perFrame:
                pos = gen_pretrain_samples(im_shape, bbox, opts.sampling.pos_perFrame * 5,
                                           opts.sampling.scale_factor, 0.1, 5, False)
                ratios = np.array([overlap_ratio(pos_, bbox) for pos_ in pos])
                pos = pos[((ratios > opts.sampling.pos_range[0]) &
                           (ratios <= opts.sampling.pos_range[1]))]
                if pos.shape[0] == 0:
                    continue
                pos = np.random.permutation(pos)[: min(pos.shape[0], opts.sampling.pos_perFrame - pos.shape[0]), :]
                pos_examples = np.vstack((pos_examples, pos.astype(dtype=np.float32)))

            neg_examples = np.zeros((0, 4), dtype=np.float32)
            while neg_examples.shape[0] < opts.sampling.neg_perFrame:
                neg = gen_pretrain_samples(im_shape, bbox, opts.sampling.neg_perFrame * 2,
                                           opts.sampling.scale_factor, 2, 10, True)
                ratios = np.array([overlap_ratio(neg_, bbox) for neg_ in neg])
                neg = neg[((ratios >= opts.sampling.neg_range[0]) &
                           (ratios < opts.sampling.neg_range[1]))]
                if neg.shape[0] == 0:
                    continue
                neg = np.random.permutation(neg)[: min(neg.shape[0], opts.sampling.neg_perFrame - neg.shape[0]), :]
                neg_examples = np.vstack((neg_examples, neg.astype(dtype=np.float32)))

            ventry['entries'].append({'im_path': entry['im_path'],
                                      'pos_boxes': pos_examples,
                                      'neg_boxes': neg_examples})
        print 'Roidb for {} finished.'.format(vkey)
        roidb.append(ventry)

    return roidb


def gen_pretrain_samples(im_shape, bbox, n, scale_factor, trans_range, scale_range, valid):
    im_h, im_w, im_c = im_shape
    im_h -= 1
    im_w -= 1

    sample = bbox.copy()
    # [center_x center_y width height]
    sample[:2] += sample[2:] / 2.
    samples = np.repeat(sample[np.newaxis, :], n, axis=0)

    samples[:, :2] += trans_range * np.hstack((bbox[2] * (np.random.rand(n, 1) * 2 - 1),
                                               bbox[3] * (np.random.rand(n, 1) * 2 - 1)))
    samples[:, 2:] *= scale_factor ** np.repeat(scale_range * (np.random.rand(n, 1) * 2 - 1), 2, axis=1)
    samples[:, 2] = np.maximum(5, np.minimum(im_w - 5, samples[:, 2]))
    samples[:, 3] = np.maximum(5, np.minimum(im_h - 5, samples[:, 3]))

    # [left top width height]
    samples = np.hstack((
        (samples[:, 0] - samples[:, 2] / 2.)[:, np.newaxis],
        (samples[:, 1] - samples[:, 3] / 2.)[:, np.newaxis],
        samples[:, 2:]
    ))

    if valid:
        samples[:, 0] = np.maximum(0, np.minimum(im_w - samples[:, 2], samples[:, 0]))
        samples[:, 1] = np.maximum(0, np.minimum(im_h - samples[:, 3], samples[:, 1]))
    else:
        samples[:, 0] = np.maximum(-samples[:, 2] / 2., np.minimum(im_w - samples[:, 2] / 2., samples[:, 0]))
        samples[:, 1] = np.maximum(-samples[:, 3] / 2., np.minimum(im_h - samples[:, 3] / 2., samples[:, 1]))

    samples = np.round(samples)
    return samples

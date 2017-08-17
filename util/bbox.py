import numpy as np
import math

def gen_samples(im, type, bbox, n, opts, trans_f, scale_f):
    """
    GEN_SAMPLES
    Generate sample bounding boxes.

    TYPE: sampling method
    'gaussian'          generate samples from a Gaussian distribution centered at bb
                        -> positive samples, target candidates
    'uniform'           generate samples from a uniform distribution around bb
                        -> negative samples
    'uniform_aspect'    generate samples from a uniform distribution around bb with varying aspect ratios
                        -> training samples for bbox regression
    'whole'             generate samples from the whole image
                        -> negative samples at the initial frame
    """
    im_h, im_w, im_c = im.shape
    im_h -= 1
    im_w -= 1
    # [center_x center_y width height]
    sample = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2., bbox[2], bbox[3]], dtype=np.float32)
    samples = np.repeat(sample[np.newaxis, :], n, axis=0)

    if type == 'gaussian':
        samples[:, :2] += trans_f * np.round(np.mean(bbox[2:])) * \
                          np.maximum(-1, np.minimum(1, 0.5 * np.random.randn(n, 2)))
        samples[:, 2:] *= np.repeat(np.array([[opts.sampling.scale_factor]]) **
                                    (scale_f * (np.random.randn(n, 1) * 2 - 1)), 2, axis=1)
    elif type == 'uniform':
        samples[:, :2] += trans_f * np.round(np.mean(bbox[2:])) * (np.random.rand(n, 2) * 2 - 1)
        samples[:, 2:] *= np.repeat(np.array([[opts.sampling.scale_factor]]) **
                                    (scale_f * (np.random.rand(n, 1) * 2 - 1)), 2, axis=1)
    elif type == 'uniform_aspect':
        samples[:, :2] += trans_f * np.repeat(bbox[2:][np.newaxis, :], n, axis=0) * (np.random.rand(n, 2) * 2 - 1)
        samples[:, 2:] *= opts.sampling.scale_factor ** (np.random.rand(n, 2) * 4 - 2)
        samples[:, 2:] *= np.repeat(np.array([[opts.sampling.scale_factor]]) **
                                    (scale_f * np.random.rand(n, 1)), 2, axis=1)
    elif type == 'whole':
        range_ = np.round([bbox[2] / 2., bbox[3] / 2., im_w - bbox[2] / 2., im_h - bbox[3] / 2.])
        stride = np.round([bbox[2] / 5., bbox[3] / 5.])
        dx, dy, ds = np.meshgrid(
            np.arange(int((range_[2] - range_[0]) / stride[0])) + range_[0],
            np.arange(int((range_[3] - range_[1]) / stride[1])) + range_[1],
            np.arange(-5, 6)
        )
        dx = dx.flatten()[:, np.newaxis]
        dy = dy.flatten()[:, np.newaxis]
        ds = ds.flatten()
        windows = np.hstack((
            dx, dy,
            (bbox[2] * opts.sampling.scale_factor ** ds)[:, np.newaxis],
            (bbox[3] * opts.sampling.scale_factor ** ds)[:, np.newaxis]
        ))

        samples = np.zeros((0, 4), dtype=np.float32)
        while samples.shape[0] < n:
            windows_ = np.random.permutation(windows)  # shuffle the windows
            samples = np.vstack((samples, windows_[:min(windows_.shape[0], n - samples.shape[0]), :]))

    samples[:, 2] = np.maximum(10, np.minimum(im_w - 10, samples[:, 2]))
    samples[:, 3] = np.maximum(10, np.minimum(im_h - 10, samples[:, 3]))

    bb_samples = np.hstack((
        (samples[:, 0] - samples[:, 2] / 2.)[:, np.newaxis],
        (samples[:, 1] - samples[:, 3] / 2.)[:, np.newaxis],
        samples[:, 2:]
    ))
    bb_samples[:, 0] = np.maximum(-bb_samples[:, 2] / 2.,
                                  np.minimum(im_w - bb_samples[:, 2] / 2, bb_samples[:, 0]))
    bb_samples[:, 1] = np.maximum(-bb_samples[:, 3] / 2.,
                                  np.minimum(im_h - bb_samples[:, 3] / 2, bb_samples[:, 1]))
    bb_samples = np.round(bb_samples)

    return bb_samples


def overlap_ratio(rect1, rect2):
    """
    Calculate overlap ratio between two bounding boxes
    :param rect1: [x, y, w, h]
    :param rect2: [x, y, w, h]
    :return: overlap ratio
    """
    box1 = rect1.copy()
    box2 = rect2.copy()
    box1[2:] += box1[:2]
    box2[2:] += box2[:2]
    ratio = 0
    iw = min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1
    if iw > 0:
        ih = min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1
        if ih > 0:
            union_area = float(rect1[2] * rect1[3] +
                               rect2[2] * rect2[3] -
                               iw * ih)
            ratio = iw * ih / union_area

    return ratio


class bbox_reg(object):

    def __init__(self, min_overlap=0.6, ld=1000, robust=0):
        self._min_overlap = min_overlap
        self._ld = ld
        self._robust = robust

        self.mu = None
        self.T = None
        self.T_inv = None
        self.Beta = None

    def train(self, X, bboxes, gt):
        """
        This class is actually after MDNet's train_bbox_regressor.m
        bboxes: list of boxes
            {
                'box'(x, y, w, h),
                'label': label,
                'overlap':overlap
            }
        gt: four-tuple(x, y, w, h)
        """
        # Get positive examples
        Y, O = self._get_examples(bboxes, gt)

        idx = np.where(O > self._min_overlap)[0]
        X = X[idx]
        Y = Y[idx]
        # add bias
        X = np.column_stack((X, np.ones(X.shape[0], dtype=np.float32)))

        # Center and decorrelate targets
        mu = np.mean(Y)
        Y = Y - mu
        S = np.dot(Y.T, Y) / Y.shape[0]
        D, V = np.linalg.eig(S)
        di = np.diag(1 / np.sqrt(D + 0.001))
        T = V.T.dot(V.dot(di))
        T_inv = V.T.dot(V.dot(np.diag(np.sqrt(D + 0.001))))
        Y = np.dot(Y, T)

        self.Beta = np.array([self._solve_robust(X, Y[:, 0], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 1], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 2], self._ld, self._robust),
                     self._solve_robust(X, Y[:, 3], self._ld, self._robust)]).T

        self.mu = mu
        self.T = T
        self.T_inv = T_inv

    def predict(self, feat, ex_boxes):
        num = ex_boxes.shape[0]
        end = self.Beta.shape[0]
        # Predict regression targets
        Y = feat.dot(self.Beta[:end-1, :]) + self.Beta[end-1, :]
        # Invert whitening transformation
        Y = Y.dot(self.T_inv) + self.mu

        # Read out predictions
        dst_ctr_x = Y[:, 0]
        dst_ctr_y = Y[:, 1]
        dst_scl_x = Y[:, 2]
        dst_scl_y = Y[:, 3]

        src_w = ex_boxes[:, 2]
        src_h = ex_boxes[:, 3]
        src_ctr_x = ex_boxes[:, 0] + 0.5 * src_w
        src_ctr_y = ex_boxes[:, 1] + 0.5 * src_h

        pred_ctr_x = (dst_ctr_x * src_w) + src_ctr_x
        pred_ctr_y = (dst_ctr_y * src_h) + src_ctr_y
        pred_w = np.exp(dst_scl_x) * src_w
        pred_h = np.exp(dst_scl_y) * src_h
        pred_boxes = np.zeros((Y.shape[0], 4))
        pred_boxes[:, 0] = (pred_ctr_x - 0.5 * pred_w)[:, 0] if num > 1 else (pred_ctr_x - 0.5 * pred_w)[0]
        pred_boxes[:, 1] = (pred_ctr_y - 0.5 * pred_h)[:, 0] if num > 1 else (pred_ctr_y - 0.5 * pred_h)[0]
        pred_boxes[:, 2] = pred_w[:, 0] if num > 1 else pred_w[0]
        pred_boxes[:, 3] = pred_h[:, 0] if num > 1 else pred_h[0]

        return pred_boxes

    def _get_examples(self, bboxes, gt):

        Y = np.zeros((0, 4), dtype=np.float32)
        O = np.zeros(0, dtype=np.float32)

        gt_w = 1. * gt[2]
        gt_h = 1. * gt[3]
        gt_ctr_x = gt[0] + 0.5 * gt_w
        gt_ctr_y = gt[1] + 0.5 * gt_h

        for bbox in bboxes:
            ex_box = bbox['box']
            ex_overlap = bbox['overlap']

            src_w = ex_box[2]
            src_h = ex_box[3]
            src_ctr_x = ex_box[0] + 0.5 * src_w
            src_ctr_y = ex_box[1] + 0.5 * src_h

            dst_ctr_x = (gt_ctr_x - src_ctr_x) * 1. / src_w
            dst_ctr_y = (gt_ctr_y - src_ctr_y) * 1. / src_h
            dst_scl_w = math.log(gt_w / src_w)
            dst_scl_h = math.log(gt_h / src_h)

            arr = [dst_ctr_x, dst_ctr_y, dst_scl_w, dst_scl_h]
            Y = np.vstack((Y, np.array(arr, dtype=np.float32)))
            O = np.hstack((O, np.array(ex_overlap, dtype=np.float32)))

        return Y, O

    def _solve_robust(self, A, y, ld, qtile):
        x, losses = self._solve(A, y, ld)
        if qtile > 0:
            pass
        return x

    def _solve(self, A, y, ld):
        y = y.reshape((y.size, 1))
        M = A.T.dot(A)
        E = np.eye(A.shape[1])
        E *= ld
        M += E
        inv_m = np.linalg.inv(M)
        x = inv_m.dot(A.T)
        x = x.dot(y)
        l = A.dot(x) - y
        x = x.reshape((1, x.size))[0]
        losses = 0.5 * np.square(l)
        return x, losses

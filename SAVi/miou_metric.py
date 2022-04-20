import torch
from torchvision import transforms
import torch.nn.functional as F
import pdb
import numpy as np
import scipy
from scipy import optimize
from tdw_dataset import TDWDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from raft_eval import EvalRAFT, viz_flow_seg

def measure_miou_metric(pred_segment, gt_segment, eval_full_res=True):
    """
    Measure the mIoU of the predicted instance segmentations for a single image
    Args:
        pred_segment: [B, H, W] torch.int32 tensor containing predicted instance segmentation id
        gt_segment: [B, H', W'] torch.int32 tensor containing ground-truth instance segmentation id (at native resolution)
        eval_full_res: evaluate the mIoU at the native resolution (i.e. the resolution of the input image)
    Returns:
        mIoU: [B,]
    """
    if eval_full_res:
        size = None
    else:
        # Downsize to the resolution of pred_segment
        size = pred_segment.shape[-2:]
        gt_segment = F.interpolate(gt_segment.float().unsqueeze(1), size=size, mode='nearest').int()

    metric = SegmentationMetrics(gt_objects=gt_segment.cpu(),
                                 pred_objects=pred_segment.int().cpu(),
                                 size=size,
                                 background_value=0)

    metric.compute_matched_IoUs(exclude_gt_ids=[0])

    return metric.mean_ious


class SegmentationMetrics(object):
    """
    A class for computing metrics given a pair of pred and gt segment maps
    """
    def __init__(
            self,
            gt_objects, # the true segmentation
            pred_objects=None, # the predicted segmentation
            background_value=0, # value of background segment
            min_gt_size=1, # num pixels needed to be a true segment
            size=None, # image size to do evaluation at
            max_objects=None
    ):
        ## attributes for all evaluations
        self.size = size
        self.background_value = background_value
        self.min_gt_size = min_gt_size
        self.max_objects = max_objects

        ## set attributes of the gt and resize
        self.gt_objects = gt_objects
        self.pred_objects = pred_objects

        ## initialize metrics
        self.best_ious = None
        self.mean_ious = None
        self.recalls = None
        self.boundary_f1_scores = None
        self.mean_boundary_f1_scores = None

    @property
    def gt_objects(self):
        return self._gt_objects
    @gt_objects.setter
    def gt_objects(self, value):
        self._set_gt_objects(value)
        self._set_gt_ids()

    @property
    def pred_objects(self):
        return self._pred_objects
    @pred_objects.setter
    def pred_objects(self, value):
        self._set_pred_objects(value)

    def _object_id_hash(self, objects, dtype_out=np.int32, val=256):
        C = objects.shape[-1]
        out = np.zeros(shape=objects.shape[:-1], dtype=dtype_out)
        for c in range(C):
            scale = np.power(val, C-1-c)
            out += scale * objects[...,c]
        return out

    def _parse_objects_tensor(self, objects):

        shape = list(objects.shape)
        if len(shape) == 2:
            objects = objects[...,None]

        dtype = objects.dtype
        if dtype == torch.uint8:
            assert (shape[-1] == 3) or (shape[-3] == 3), shape
            channels_last = True if shape[-1] == 3 else False
        else:
            assert dtype == torch.int32, dtype
            if (shape[-1] == 1) or (shape[-3] == 1):
                channels_last = True if shape[-1] == 1 else False
            else: # 3 channels
                objects = objects[...,None]
                channels_last = True
                shape = shape + [1]

        self._temporal = False
        if len(shape) == 3:
            objects = objects[None]
            self.B = 1
            self.T = 1
            self.BT = self.B
        elif len(shape) == 5:
            self._temporal = True
            self.B, self.T = shape[:2]
            self.BT = self.B*self.T
            objects = objects.view(self.BT,*shape[2:])
        else:
            assert len(objects.shape) == 4, "torch objects must have shape [BT,C,H,W] or [BT,H,W,C]"
            self.B = shape[0]
            self.T = 1
            self.BT = self.B

        if self.max_objects is None:
            if dtype == torch.uint8:
                hashed = object_id_hash(objects, channels_last=channels_last)
            else:
                hashed = objects
            ims = list(hashed)
            num_objects = [int(torch.unique(im).size(0)) for im in ims]
            self.max_objects = max(num_objects)

        if dtype == torch.uint8:
            objects = object_id_hash(objects, channels_last=channels_last)

        if not channels_last:
            objects = objects.permute(0,2,3,1)

        if self.size is not None:
            objects = F.interpolate(objects.permute(0,3,1,2).float(), size=self.size, mode='nearest').permute(0,2,3,1).int()

        assert objects.dtype == torch.int32, objects.dtype
        return objects.numpy()

    def _parse_objects_array(self, objects):
        if objects.shape[-1] not in [1,3]:
            objects = objects[...,None]
        if objects.shape[-1] == 3:
            assert objects.dtype == np.uint8, objects.dtype
            objects = self._object_id_hash(objects)
        else:
            assert objects.dtype == np.int32

        self._temporal = False
        if len(objects.shape) == 5:
            self._temporal = True
            self.B,self.T = objects.shape[:2]
            self.BT = self.B*self.T
            objects = objects.reshape([self.BT] + objects.shape[2:])
        elif len(objects.shape) == 3:
            self.B = objects.shape[0]
            self.T = 1
            self.BT = self.B
            objects = objects[...,None]
        else:
            assert len(objects.shape) == 4, objects.shape
            self.B = objects.shape[0]
            self.T = 1
            self.BT = self.B

        if self.size is not None:
            objects = map(lambda im: skimage.transform.resize(im.astype(float), self.size, order=0).astype(np.int32), [objects[ex] for ex in range(self.BT)])
            objects = np.stack(objects, 0)

    def _set_gt_objects(self, objects):
        if isinstance(objects, torch.Tensor):
            objects = self._parse_objects_tensor(objects)
        else:
            objects = self._parse_objects_array(objects)

        assert len(objects.shape) == 4, objects.shape
        assert objects.shape[-1] == 1, objects.shape
        assert objects.dtype == np.int32, objects.dtype

        self._gt_objects = objects[...,0]
        self.gt_shape = self._gt_objects.shape
        self.size = self.gt_shape[-2:]

    def _set_gt_ids(self):
        self.gt_ids = []
        for ex in range(self.BT):
            self.gt_ids.append(
                np.unique(self.gt_objects[ex]))


    def _set_pred_objects(self, objects):
        if objects is None:
            return
        if isinstance(objects, torch.Tensor):
            objects = self._parse_objects_tensor(objects)
        else:
            objects = self._parse_objects_array(objects)

        assert len(objects.shape) == 4, objects.shape
        assert objects.shape[-1] == 1, objects.shape
        assert objects.dtype == np.int32, objects.dtype

        ## subtract off the minimum value
        offsets = objects.min(axis=(1,2), keepdims=True)
        objects -= offsets

        self._pred_objects = objects[...,0]


    def _get_mask(self, objects, obj_id=0):
        return objects == obj_id

    def get_gt_mask(self, ex, t=0, obj_id=0):
        b = ex*self.T + t
        return self._get_mask(self.gt_objects[b], obj_id)

    def get_pred_mask(self, ex, t=0, obj_id=0):
        assert self.pred_objects is not None
        b = ex*self.T + t
        return self._get_mask(self.pred_objects[b], obj_id)

    def get_background(self, ex, t=0):
        return self.get_gt_mask(ex, t, self.background_value)

    def _mask_metrics(self):
        return {'iou': self.mask_IoU}

    @staticmethod
    def mask_IoU(pred_mask, gt_mask, min_gt_size=1):
        """Compute intersection over union of two boolean masks"""
        assert pred_mask.shape == gt_mask.shape, (pred_mask.shape, gt_mask.shape)
        assert pred_mask.dtype == gt_mask.dtype == bool, (pred_mask.dtype, gt_mask.dtype)
        num_gt_px = gt_mask.sum()
        num_pred_px = pred_mask.sum()
        if num_gt_px < min_gt_size:
            return np.nan

        overlap = (pred_mask & gt_mask).sum().astype(float)
        IoU = overlap / (num_gt_px + num_pred_px - overlap)
        return IoU

    def compute_matched_IoUs(self, pred_objects=None, exclude_gt_ids=[], metric='iou'):
        if pred_objects is not None:
            self.pred_objects = pred_objects

        assert metric in ['iou', 'precision', 'recall'], "Metric must be 'iou', 'precision', or 'recall'"
        metric_func = self._mask_metrics()[metric]

        exclude_ids = list(set(exclude_gt_ids + [self.background_value]))
        best_IoUs = []
        best_pred_objs = []
        matched_preds, matched_gts = [], []

        for b in range(self.BT):

            ex, t = (b // self.T, b % self.T)

            # the ids in each gt mask
            ids_here = [o for o in self.gt_ids[b] if o not in exclude_ids]
            num_gt = len(ids_here)

            # the pred masks
            preds = map(lambda o_id: self.get_pred_mask(ex, t, o_id),
                        sorted(list(np.unique(self.pred_objects[b]))))
            preds = list(preds)
            num_preds = len(preds)

            # compute full matrix of ious
            gts = []
            ious = np.zeros((num_gt, num_preds), dtype=np.float32)
            for m in range(num_gt):
                gt_mask = self.get_gt_mask(ex, t, ids_here[m])
                gts.append(gt_mask)
                for n in range(num_preds):
                    pred_mask = preds[n]
                    # pdb.set_trace()
                    iou = metric_func(pred_mask, gt_mask, self.min_gt_size)
                    ious[m,n] = iou if not np.isnan(iou) else 0.0

            # linear assignment
            gt_inds, pred_inds = scipy.optimize.linear_sum_assignment(1.0 - ious)

            # output values
            best = np.array([0.0] * len(ids_here))
            best[gt_inds] = ious[gt_inds, pred_inds]
            best_IoUs.append(best)
            best_objs = np.array([0] * len(ids_here))
            best_objs[gt_inds] = np.array([sorted(list(np.unique(self.pred_objects[b])))[i] for i in pred_inds])
            best_pred_objs.append(best_objs)

            count = 0
            matched_pred = []
            for m in range(num_gt):
                if count < len(gt_inds):
                    if m == gt_inds[count]:
                        matched_pred.append(preds[pred_inds[count]])
                        count += 1
                        continue

                matched_pred.append(np.zeros_like(preds[0]))
            matched_preds.append(matched_pred)

        self.best_ious = best_IoUs
        self.best_object_ids = best_pred_objs
        self.seg_out = (matched_preds, [gts], best_IoUs)

        return self.mean_ious

    @property
    def mean_ious(self):
        if self.best_ious is None:
            return None
        elif self._mean_ious is None:
            self._mean_ious = np.array([np.nanmean(self.best_ious[b]) for b in range(self.BT)])
            if self._temporal:
                self._mean_ious = self._mean_ious.reshape((self.B, self.T))
            return self._mean_ious
        else:
            return self._mean_ious
    @mean_ious.setter
    def mean_ious(self, value=None):
        if value is not None:
            raise ValueError("You can't set the mean ious, you need to compute it")
        self._mean_ious = value

if __name__ == "__main__":
    dataset_dir = '/data2/honglinc/playroom_large_v3_images'
    batch_size = 1
    ckpt_path = '../projects/Panoptic-DeepLab/RAFT/models/raft-sintel.pth'
    flow_threshold = 0.5  # Important: this threshold is used for TDW

    val_dataset = TDWDataset(dataset_dir, training=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    raft_model = EvalRAFT(ckpt_path, flow_threshold)

    image_1, image_2, segment_map, gt_moving = next(iter(val_dataloader))
    # Note: images should NOT be normalized before passing to raft
    image_1, image_2 = image_1.cuda(), image_2.cuda()
    flow, magnitude, motion_segment = raft_model(image_1, image_2)

    # measuring the mIoU of the MOTION segmentations
    moving_obj_miou = measure_miou_metric(motion_segment, gt_moving.int())
    print('mIoU (motion segmentations): ', moving_obj_miou.mean())


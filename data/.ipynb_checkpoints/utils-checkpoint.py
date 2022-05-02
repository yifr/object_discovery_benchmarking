import os
import numpy as np
import copy
from PIL import Image
import torch
from torchvision import transforms
import kornia
import cv2
import torch
import torch.nn as nn


PASSES_DICT = {
    "images": "_img",
    "objects": "_id",
    "flows": "_flow",
    "depths": "_depth",
    "normals": "_normals",
    "categories": "_category",
    "albedos": "_albedo"
}

def depth_uint8_to_float32(depthmap, normalization=100.1, channels_last=False):
    assert depthmap.dtype == torch.uint8, depthmap.dtype
    shape = depthmap.shape
    if channels_last:
        H,W,C = shape[-3:]
    else:
        C,H,W = shape[-3:]

    assert (C == 3), "Depthmap must have 3 channels but has %d" % C
    channel_weights = torch.tensor([256.0*256.0, 256.0, 1.0], dtype=torch.float32)

    channel_weights = torch.reshape(
        channel_weights,
        ([1] * (len(shape)-3)) + ([1,1,C] if channels_last else [C,1,1]))

    out = torch.sum(depthmap * channel_weights,
                    dim=(-1 if channels_last else -3),
                    keepdim=True)
    out *= torch.tensor(normalization / (256.0**3), dtype=torch.float32)
    return out


def object_id_hash(objects, dtype_out=torch.int32, val=256, channels_last=False):
    '''
    objects: [...,C]
    val: a number castable to dtype_out

    returns:
    out: [...,1] where each value is given by sum([val**(C-1-c) * objects[...,c:c+1] for c in range(C)])
    '''
    if not isinstance(objects, torch.Tensor):
        objects = torch.tensor(objects)
    if not channels_last:
        objects = objects.permute(0,2,3,1)
    C = objects.shape[-1]
    val = torch.tensor(val, dtype=dtype_out)
    objects = objects.to(dtype_out)
    out = torch.zeros_like(objects[...,0:1])
    for c in range(C):
        scale = torch.pow(val, C-1-c)
        out += scale * objects[...,c:c+1]
    if not channels_last:
        out = out.permute(0,3,1,2)

    return out

def rgb_to_xy_flows(flows, to_image_coordinates=True, to_sampling_grid=False):
    assert flows.dtype == torch.uint8, flows.dtype
    assert flows.shape[-3] == 3, flows.shape
    flows_hsv = kornia.color.rgb_to_hsv(flows.float() / 255.)

    hue, sat, val = flows_hsv.split([1, 1, 1], dim=-3)
    flow_x = torch.cos(hue) * val
    flow_y = torch.sin(hue) * val

    if to_image_coordinates:
        flow_h = -flow_y
        flow_w = flow_x
        return torch.cat([flow_h, flow_w], -3)
    elif to_sampling_grid:
        return torch.cat([flow_x, -flow_y], -3)
    else:
        return torch.cat([flow_x, flow_y], -3)

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

class RgbFlowToXY(object):
    def __init__(self, to_image_coordinates=True, to_sampling_grid=False):
        self.to_image_coordinates = to_image_coordinates
        self.to_sampling_grid = to_sampling_grid
    def __call__(self, flows_rgb):
        return rgb_to_xy_flows(flows_rgb, self.to_image_coordinates, self.to_sampling_grid)

class FlowToRgb(object):

    def __init__(self, max_speed=1.0, from_image_cooordinates=True, from_sampling_grid=False):
        self.max_speed = max_speed
        self.from_image_cooordinates = from_image_cooordinates
        self.from_sampling_grid = from_sampling_grid

    def __call__(self, flow):
        assert flow.size(-3) == 2, flow.shape
        if self.from_sampling_grid:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
            flow_y = -flow_y
        elif not self.from_image_cooordinates:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
        else:
            flow_h, flow_w = torch.split(flow, [1,1], dim=-3)
            flow_x, flow_y = [flow_w, -flow_h]

        angle = torch.atan2(flow_y, flow_x) # in radians from -pi to pi
        speed = torch.sqrt(flow_x**2 + flow_y**2) / self.max_speed

        hue = torch.fmod(angle, torch.tensor(2 * np.pi))
        sat = torch.ones_like(hue)
        val = speed

        hsv = torch.cat([hue, sat, val], -3)
        rgb = kornia.color.hsv_to_rgb(hsv)
        return rgb

class OpticalFlowRgbTo2d(object):

    def __init__(self, channels_last=False, max_speed=1.0, to_image_coordinates=True):
        self.channels_last = channels_last
        self.max_speed = max_speed
        self.to_image_coordinates = to_image_coordinates

    @staticmethod
    def hsv_to_2d_velocities_and_speed(hsv, max_speed=1.0, to_image_coordinates=False):
        if hsv.dtype == np.uint8:
            hsv = hsv / 255.0
        h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
        ang = h * 2 * np.pi
        speed = v * max_speed
        flow_x = np.cos(ang) * speed
        flow_y = np.sin(ang) * speed
        mag = np.sqrt(flow_x**2 + flow_y**2)

        if to_image_coordinates:
            flow = np.stack([-flow_y, flow_x, mag], -1)
        else:
            flow = np.stack([flow_x, flow_y, mag], -1)

        return flow

    def __call__(self, rgb_flows):

        if rgb_flows.dtype == np.float32:
            rgb_flows = np.clip(rgb_flows * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            assert rgb_flows.dtype == np.uint8

        H,W,C = rgb_flows.shape
        out = np.zeros((H,W,3), dtype=np.float32)
        hsv = cv2.cvtColor(rgb_flows, cv2.COLOR_RGB2HSV)
        velocities = self.hsv_to_2d_velocities_and_speed(
            hsv,
            max_speed=self.max_speed,
            to_image_coordinates=self.to_image_coordinates
        )
        out = velocities
        return out
class RgbToIntSegments(object):

    def __init__(self, channels_last=False):
        self.channels_last = channels_last

    def __call__(self, objects):
        if len(objects.shape) == 3:
            objects = objects[None]
        return object_id_hash(objects, channels_last=self.channels_last)


class RgbToOneHotSegments(object):

    def __init__(self, channels_last=False, max_objects=8):
        self.channels_last = channels_last
        self.max_objects = max_objects

    @staticmethod
    def segments_rgb_to_one_hot(objects, channels_last=False, max_objects=8):
        if len(objects.shape) == 3:
            objects = objects[None]
        hashed_objects = object_id_hash(objects, channels_last=channels_last)
        if channels_last:
            hashed_objects = hashed_objects.permute(0, 3, 1, 2) # [B,1,H,W]
        hashed_objects = list(hashed_objects[:,0]) # B-length list of [H,W]
        out = []
        for b,im in enumerate(hashed_objects):
            one_hot_im = []
            obj_ids = torch.unique(im).to(im)
            for o_id in list(obj_ids[:max_objects]):
                obj = (im == o_id)
                one_hot_im.append(obj)
            one_hot_im = torch.stack(one_hot_im, 0).to(torch.float32)
            pad = torch.zeros(size=([max_objects - int(one_hot_im.shape[0])] + list(one_hot_im.shape)[1:])).to(one_hot_im)
            one_hot_im = torch.cat([one_hot_im, pad], 0)
            out.append(one_hot_im)
        out = torch.stack(out, 0)
        return out

    def __call__(self, objs):
        return self.segments_rgb_to_one_hot(objs, self.channels_last, self.max_objects)

class ToTensor(object):

    def __init__(self, to_float=False):
        self.to_float = to_float

    def __call__(self, arr):
        if not isinstance(arr, np.ndarray):
            return transforms.ToTensor()(arr)
        elif (len(arr.shape) != 3):
            return torch.from_numpy(arr)
        elif self.to_float:
            return transforms.ToTensor()(arr)
        else:
            return torch.from_numpy(arr.transpose((2,0,1)))

class ToTensorMovie(object):

    def __init__(self, to_float=False, channels_last=False):
        self.to_float = to_float
        self.channels_last = channels_last

    def __call__(self, arr):
        assert len(arr.shape) == 4, "arr must be a [T,H,W,C] numpy array but has shape %s" % arr.shape
        if self.to_float:
            out = torch.stack([
                transforms.ToTensor()(im) for im in list(arr)], 0)
            if self.channels_last:
                out = out.permute(0,2,3,1)
            return out
        else:
            out = torch.from_numpy(arr)
            if not self.channels_last:
                out = out.permute(0,3,1,2)
            return out

class DeltaImages(object):

    def __init__(self, thresh=None, normalization=255.0, channels_last=False):
        self.thresh = thresh
        self.normalization = normalization
        self.channels_last = channels_last

    def __call__(self, ims):
        assert len(ims.shape) == 4, (ims.shape)

        if ims.shape[0] == 1:
            delta_images = np.zeros(list(ims.shape[:-1]) + [1],
                                    dtype=(bool if self.thresh else np.float32))
            if not self.channels_last:
                delta_images = delta_images.transpose((0,3,1,2))
            return delta_images

        if ims.dtype == np.uint8:
            ims = ims.astype(np.float32) / self.normalization
        else:
            assert ims.dtype == np.float32

        intensities = ims.mean(axis=-1, keepdims=True)
        diffs = intensities[1:,...] - intensities[:-1,...]
        diffs = np.abs(diffs)
        delta_images = np.concatenate(
            [np.zeros_like(diffs[0:1]), diffs],
            axis=0)
        if self.thresh is not None:
            delta_images = (delta_images > self.thresh).astype(np.float32)

        if not self.channels_last:
            delta_images = delta_images.transpose((0,3,1,2))

        return delta_images

class OpticalFlow(object):

    default_farneback_kwargs = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    def __init__(self, thresh=None, channels_last=False, to_rgb=False, **kwargs):
        self.thresh = thresh
        self.channels_last = channels_last
        self.to_rgb = to_rgb
        self.flow_kwargs = copy.deepcopy(self.default_farneback_kwargs)
        self.flow_kwargs.update(kwargs)

    @staticmethod
    def flow_to_rgb(flow):
        """convert flow to an rgb image where hue indicates angle, value indicate speed"""
        assert len(flow.shape) == 3 and flow.shape[-1] == 2, flow.shape
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        h = ang*180/np.pi/2
        s = 255*np.ones_like(h)
        v = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = np.stack([h,s,v], -1)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        return rgb

    def __call__(self, rgb_mov):
        assert len(rgb_mov.shape) == 4, (rgb_mov.shape)

        if rgb_mov.shape[0] == 1:
            return np.zeros(rgb_mov.shape, dtype=np.float32)

        ## opencv needs to start from uint8?
        if rgb_mov.dtype == np.float32:
            rgb_mov = (np.clip(rgb_mov * 255.0, 0.0, 255.0)).astype(np.uint8)
        else:
            assert rgb_mov.dtype == np.uint8

        T,H,W,C = rgb_mov.shape
        out_channels = 3 if self.to_rgb else 2
        prev_ims = rgb_mov[0:-1]
        next_ims = rgb_mov[1:]
        out = np.zeros((T,H,W,out_channels), dtype=np.float32)
        for t in range(T-1):
            prev = cv2.cvtColor(prev_ims[t], cv2.COLOR_RGB2GRAY)
            nxt = cv2.cvtColor(next_ims[t], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, **self.flow_kwargs)
            out[t+1] = flow if not self.to_rgb else self.flow_to_rgb(flow)

        if self.thresh is not None:
            out = (np.abs(out).sum(-1, keepdims=True) > self.thresh).astype(np.float32)

        if not self.channels_last:
            out = out.transpose((0,3,1,2))

        return out

class FillContours(object):

    def __init__(self, to_float=False, channels_last=False):
        self.to_float = to_float
        self.channels_last = channels_last

    @staticmethod
    def fill_contours(bin_img):
        """Use opencv to detect and fill the contours of a binary image"""
        img = np.copy(bin_img)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        c_max = []
        max_area = 0
        max_cnt = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)

            ## find max contour
            if (area > max_area):
                if (max_area != 0):
                    c_min = []
                    c_min.append(max_cnt)
                    cv2.drawContours(img, c_min, -1, (0,0,0), cv2.FILLED)
                max_area = area
                max_cnt = cnt
            else:
                c_min = []
                c_min.append(cnt)
                cv2.drawContours(img, c_min, -1, (0,0,0), cv2.FILLED)

        c_max.append(max_cnt)
        if isinstance(max_cnt, int):
            return img
        else:
            cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
            return img

    def __call__(self, bin_img):
        shape = bin_img.shape
        if len(shape) == 3:
            bin_img = bin_img[None]
            T = 1
            H,W,C = shape
        else:
            assert len(shape) == 4, shape
            T,H,W,C = shape

        ## required by opencv2
        bin_img = bin_img.astype(np.uint8)

        out = np.zeros((T,H,W,1), dtype=bool)
        for t in range(T):
            filled = self.fill_contours(bin_img[t])
            out[t] = (filled > 0)

        if not self.channels_last:
            out = out.transpose((0,3,1,2))

        if self.to_float:
            out = out.astype(np.float32)

        if len(shape) == 3:
            out = out[0]

        return out

class FillDeltaImages(nn.Module):

    def __init__(self, thresh=0.01):
        super().__init__()
        self.thresh = thresh
        self.delta_fn = DeltaImages(thresh=self.thresh)
        self.fill_fn = FillContours(to_float=True)

    def _to_numpy(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.

        x = x.permute(0,2,3,1)
        x = x.detach().cpu().numpy()
        return x

    def forward(self, video):

        device = video.device
        if isinstance(video, torch.Tensor):
            video = self._to_numpy(video)

        deltas = self.delta_fn(video)
        filled = self.fill_fn(deltas[1:,0,:,:,None])
        return torch.tensor(filled, device=device)

class FillDeltaImagesRaft(FillDeltaImages):

    def forward(self, img1, img2, **kwargs):

        video = torch.cat([img1, img2], 0) / 255.0
        filled = super().forward(video)
        return None, filled

if __name__ == '__main__':
    depth = torch.ones(size=[4,3,256,256], dtype=torch.uint8)
    depth = depth_uint8_to_float32(depth, channels_last=False)
    print(depth.shape, depth.dtype)
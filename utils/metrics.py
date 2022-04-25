import torch
import torch.nn.functional as F

# Taken from gist: https://gist.githubusercontent.com/vadimkantorov/bd1616a3a9eea89658ea3efb1f9a1d5d


def adjusted_rand_index(true_mask, pred_mask, foreground=True):
    """
    Provides an implementation of the Adjusted Rand Index. Ignores points with no cluster label
    in `true_mask` (ie; those points where `true_mask` is all zero). This means it provides a
    "foreground" segmentation metric
    Args:
        true_mask: tensor or np.array of size (batch, max_objects, time, channel, height, width)
        pred_mask: predicted tensor of same size
    """
    """
    B, max_num_entities, T, C, H, W = true_mask.shape
    desired_shape = (B, T*C*H*W, max_num_entities)
    true_mask = true_mask.reshape(desired_shape)
    pred_mask = pred_mask.reshape(desired_shape)
    """
    if foreground:
        # Assumes first group is background pixels
        true_mask = true_mask[..., 1:]

    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    assert not (
        n_points <= n_true_groups and n_points <= n_pred_groups
    ), "adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint."

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32)  # One hot encoding
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)
    nij = torch.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    def _all_equal(values): return torch.all(
        torch.eq(values, values[..., :1]), dim=-1)
    both_single_cluster = torch.logical_and(
        _all_equal(true_group_ids), _all_equal(pred_group_ids)
    )
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def mean_IOU(true_mask, pred_mask, eps=1e-8):
    """
    Implements Mean Intersection over Union (IOU) metric.
    Args:
        true_mask: tensor: B x T x C x H x W
        pred_mask: tensor: B x T x C x H x W
    Returns
        mean IOU score over whole batch
    """
    B, T, C, H, W = true_mask.shape

    true_masks = true_mask.reshape(B, -1).to(torch.uint8)
    pred_masks = pred_mask.reshape(B, -1).to(torch.uint8)

    intersection = true_masks & pred_masks
    union = true_masks | pred_masks
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


if __name__ == "__main__":
    B, T, C, H, W = 1, 10, 3, 128, 128
    t1 = torch.rand((B, T, C, H, W))
    t2 = torch.rand((B, T, C, H, W))
    t1_b = t1.clone()
    t1_b[0][:5] += 10
    miou_diff = mean_IOU(t1, t2)
    miou_half = mean_IOU(t1, t1_b)
    miou_same = mean_IOU(t1, t1)
    print("diff", miou_diff, "same", miou_same, "partial", miou_half)

import numpy as np
import cv2
import torch
from typing import Union
from torchvision import transforms

from metrics.edge_metrics import get_ods_ap
from metrics.depth_map_metrics import compute_metrics

from kornia.filters import canny, box_blur


def get_edge_metrics(pred_batch_images, validation_image_tensors):
    with torch.inference_mode():
        pred_batch_images_edges = canny(pred_batch_images, 0.1, 0.25, kernel_size=(3,3))[1]
        pred_batch_images_edges_blured = box_blur(pred_batch_images_edges, kernel_size=(5,5))

        # pil_edge = transforms.functional.to_pil_image(pred_batch_images_edges[0].squeeze())
        # pil_edge.save('kornia_edge_0.jpg')

    avg_img_ap = 0
    avg_img_ods = 0

    avg_img_ap_blur = 0
    avg_img_ods_blur = 0

    for i in range(validation_image_tensors.shape[0]):
        pred_image_edge = pred_batch_images_edges[i].squeeze().cpu().numpy()
        condition_img  = validation_image_tensors[i].cpu().numpy()

        pred_image_edge_blur = pred_batch_images_edges_blured[i].squeeze().cpu().numpy()
        
        assert np.unique(condition_img)[0] == 0 and np.unique(condition_img)[1] == 1, "Condition edge image should have only 0 and 1 values"
        ods, ap = get_ods_ap(img_pred=pred_image_edge, img_true=condition_img, is_uint8=False)
        ods_blur, ap_blur = get_ods_ap(img_pred=pred_image_edge_blur, img_true=condition_img, is_uint8=False)

        avg_img_ap += ap
        avg_img_ods += ods

        avg_img_ap_blur += ap_blur
        avg_img_ods_blur += ods_blur

    avg_img_ap = avg_img_ap / validation_image_tensors.shape[0]
    avg_img_ods = avg_img_ods / validation_image_tensors.shape[0]

    avg_img_ap_blur = avg_img_ap_blur / validation_image_tensors.shape[0]
    avg_img_ods_blur = avg_img_ods_blur / validation_image_tensors.shape[0]

    return  avg_img_ods, avg_img_ap, \
            avg_img_ods_blur, avg_img_ap_blur, \
            pred_batch_images_edges, pred_batch_images_edges_blured


def torch_percentile(t: torch.tensor, q: float, dim=-1) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.size(dim) - 1))
    result = t.kthvalue(k, dim=dim).values
    return result

def colorize_batched(value, vmin=None, vmax=None,invalid_val=-99, invalid_mask=None, background_color=(128), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    #if isinstance(value, torch.Tensor):
    #    value = value.detach().cpu().numpy()

    b,h,w,ch = value.shape
    #value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = ~invalid_mask

    # normalize
    value[invalid_mask] = np.nan
    vmin = torch_percentile(value.reshape(b,-1),2, dim=-1) if vmin is None else vmin
    vmax = torch_percentile(value.reshape(b,-1),85,dim=-1) if vmax is None else vmax
    #print (value.shape)
    if all(vmin != vmax):
        value = (value - vmin.reshape(-1, 1, 1, 1)) / (vmax - vmin).reshape(-1,1,  1, 1)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values
    #print (value.shape)
    value[invalid_mask] = np.nan
    #cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    #value = cmapper(value, bytes=True)  # (nxmx4)
    value = (255*value).clamp(0,255).byte()
    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = torch.power(img, 2.2)
        img = img * 255
        img = img.byte()#.astype(np.uint8)
    return 255 - img


def get_depth_metrics(pred_batch_images, validation_target_tensors, zoe_depth_model):
    """
    condition_img: target PIL image (NOT colorized depth map)
    """

    avg_metrics_dict = dict(a1=0, a2=0, a3=0, abs_rel=0, rmse=0, log_10=0, rmse_log=0, silog=0, sq_rel=0)
    
    with torch.inference_mode():
        gt_depthes = zoe_depth_model.infer(validation_target_tensors, with_flip_aug=False)
        pred_depthes = zoe_depth_model.infer(pred_batch_images, with_flip_aug=False)
          
        pred_depth_images = colorize_batched(pred_depthes)

        for i in range(pred_depth_images.shape[0]):
            pred_depth = pred_depthes[i]
            gt_depth = gt_depthes[i]

            metrics_dict = compute_metrics(gt=gt_depth, pred=pred_depth)
            for metric_name, metric_value in metrics_dict.items():
                avg_metrics_dict[metric_name] += metric_value

    for key, value in avg_metrics_dict.items():
        avg_metrics_dict[key] = value / pred_depthes.shape[0]

    del gt_depthes
    del pred_depthes
    torch.cuda.empty_cache()

    return avg_metrics_dict, pred_depth_images

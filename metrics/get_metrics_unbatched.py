import numpy as np
import cv2
import torch
from typing import Union
from torchvision import transforms

from metrics.edge_metrics import get_ods_ap
from metrics.depth_map_metrics import compute_metrics


def get_edge_metrics(pred_images, condition_img):
    t_lower = 100  # Lower Threshold 
    t_upper = 200  # Upper threshold 
    condition_img = np.array(condition_img.convert("L"))

    avg_img_ap = 0
    avg_img_ods = 0

    avg_img_ap_blur = 0
    avg_img_ods_blur = 0

    condition_img_blur = cv2.blur(condition_img,(5,5))

    for pred_image in pred_images:
        pred_image = np.array(pred_image)
        pred_image_edge = cv2.Canny(pred_image, t_lower, t_upper)
        pred_image_edge_blur = cv2.blur(pred_image_edge,(5,5))

        ods, ap = get_ods_ap(img_pred=pred_image_edge, img_true=condition_img, is_uint8=True)
        ods_blur, ap_blur = get_ods_ap(img_pred=pred_image_edge_blur, img_true=condition_img_blur, is_uint8=True)

        avg_img_ap += ap
        avg_img_ods += ods

        avg_img_ap_blur += ap_blur
        avg_img_ods_blur += ods_blur

    avg_img_ap = avg_img_ap / len(pred_images)
    avg_img_ods = avg_img_ods / len(pred_images)

    avg_img_ap_blur = avg_img_ap_blur / len(pred_images)
    avg_img_ods_blur = avg_img_ods_blur / len(pred_images)

    last_pred_image_edge = pred_image_edge
    last_pred_image_edge_blur = pred_image_edge_blur

    return  avg_img_ods, avg_img_ap, \
            avg_img_ods_blur, avg_img_ap_blur, \
            last_pred_image_edge, last_pred_image_edge_blur, condition_img_blur


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
    return 255 - img.detach().cpu().numpy()


def get_depth_metrics(pred_images, condition_img, zoe_depth_model, weight_dtype):
    """
    condition_img: target PIL image (NOT colorized depth map)
    """

    avg_metrics_dict = dict(a1=0, a2=0, a3=0, abs_rel=0, rmse=0, log_10=0, rmse_log=0, silog=0, sq_rel=0)

    for i, pred_image in enumerate(pred_images):
        with torch.inference_mode():
            condition_img_tensor = transforms.ToTensor()(condition_img).unsqueeze(0).to(zoe_depth_model.device,
                                                                                        dtype=weight_dtype)
            pred_image_tensor = transforms.ToTensor()(pred_image).unsqueeze(0).to(zoe_depth_model.device,
                                                                                        dtype=weight_dtype)
            gt_depth = zoe_depth_model.infer(condition_img_tensor, with_flip_aug=False)
            pred_depth = zoe_depth_model.infer(pred_image_tensor, with_flip_aug=False)

            if i == len(pred_images) - 1:
                pred_depth_image = colorize_batched(pred_depth)
                pred_depth_image = pred_depth_image.squeeze()


        metrics_dict = compute_metrics(gt=gt_depth, pred=pred_depth)
        for metric_name, metric_value in metrics_dict.items():
            avg_metrics_dict[metric_name] += metric_value

    for key, value in avg_metrics_dict.items():
        avg_metrics_dict[key] = value / len(pred_images)

    return avg_metrics_dict, pred_depth_image



def get_color_metric(pred_images, target_img):
    target_image = np.array(target_img).astype(float) / 255.0

    l2_norms = 0

    for pred_image in pred_images:
        pred_image = np.array(pred_image).astype(float) / 255.0
        l2_norm = np.linalg.norm(target_image - pred_image)
        l2_norms += l2_norm

    l2_norm_avg = l2_norms / len(pred_images)
    return l2_norm_avg

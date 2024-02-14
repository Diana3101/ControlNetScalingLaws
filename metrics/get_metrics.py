import numpy as np
import cv2

from metrics.edge_metrics import get_ods_ap

def get_edge_metric(pred_images, condition_img):
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

        ods, ap = get_ods_ap(img_pred=pred_image_edge, img_true=condition_img)
        ods_blur, ap_blur = get_ods_ap(img_pred=pred_image_edge_blur, img_true=condition_img_blur)

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



def get_color_metric(pred_images, target_img):
    target_image = np.array(target_img).astype(float) / 255.0

    l2_norms = 0

    for pred_image in pred_images:
        pred_image = np.array(pred_image).astype(float) / 255.0
        l2_norm = np.linalg.norm(target_image - pred_image)
        l2_norms += l2_norm

    l2_norm_avg = l2_norms / len(pred_images)
    return l2_norm_avg

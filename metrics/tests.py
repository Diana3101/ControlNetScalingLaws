import numpy as np
from PIL import Image
import cv2
import pandas as pd

from metrics.edge_metrics import get_ods_ap

def get_edge_metric(pred_image, condition_img):
    t_lower = 100  # Lower Threshold 
    t_upper = 200  # Upper threshold 

    condition_img = np.array(condition_img.convert("L"))

    avg_img_ap = 0
    avg_img_ods = 0

    avg_img_ap_blur = 0
    avg_img_ods_blur = 0

    condition_img_blur = cv2.blur(condition_img,(5,5))

    # for pred_image in pred_images:
    pred_image = np.array(pred_image)
    pred_image_edge = cv2.Canny(pred_image, t_lower, t_upper)
    pred_image_edge_blur = cv2.blur(pred_image_edge,(5,5))

    ods, ap = get_ods_ap(img_pred=pred_image_edge, img_true=condition_img)
    ods_blur, ap_blur = get_ods_ap(img_pred=pred_image_edge_blur, img_true=condition_img_blur)

    avg_img_ap += ap
    avg_img_ods += ods

    avg_img_ap_blur += ap_blur
    avg_img_ods_blur += ods_blur

    # avg_img_ap = avg_img_ap / len(pred_images)
    # avg_img_ods = avg_img_ods / len(pred_images)

    # avg_img_ap_blur = avg_img_ap_blur / len(pred_images)
    # avg_img_ods_blur = avg_img_ods_blur / len(pred_images)

    last_pred_image_edge = pred_image_edge
    last_pred_image_edge_blur = pred_image_edge_blur

    return  avg_img_ods, avg_img_ap, \
            avg_img_ods_blur, avg_img_ap_blur, \
            last_pred_image_edge, last_pred_image_edge_blur, condition_img_blur



pred_image = Image.open('tests_data/0-target.png')
condition_img = Image.open('tests_data/0-condition.png')
avg_img_ods, avg_img_ap, avg_img_ods_blur, avg_img_ap_blur, _, _, _ = get_edge_metric(pred_image=pred_image, 
                condition_img=condition_img)

# Metrics in ideal case (target image from the fill50k dataset)
print(avg_img_ods, avg_img_ap, avg_img_ods_blur, avg_img_ap_blur)


pred = Image.open('tests_data/pred_val_1.png').convert("RGB")
pred_array = np.array(pred).astype(float) / 255.0

true = Image.open('validation_set/target_image_1.png')
true_array = np.array(true).astype(float) / 255.0

dist = np.linalg.norm(true_array - pred_array)
print(dist)
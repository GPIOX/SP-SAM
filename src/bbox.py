import cv2
import numpy as np

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图
def get_min_area_rect_from_mask(mask):
    return mask_find_bboxs(mask)

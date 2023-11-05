import torch
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.bounding_box import BoxList
import math
import numpy as np
import json
import random
import colorsys
from PIL import Image
from matplotlib import patches,  lines
import IPython.display
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
import cv2 as cv
import os
import argparse
from maskrcnn_benchmark.config import cfg

# box (big, mid, small), (horizontal, vertical, central)
# position sub->obj (top, bottom, left, right, top_left) distance (big, mid, small)
def generate_box(size, shape):
    if shape == 'horizontal':
        w, h = 1, 0.25
    elif shape == 'vertical':
        w, h = 0.25, 1
    elif shape == 'central':
        w, h = 0.5, 0.5

    if size == 'big':
        d = 1
    elif size == 'mid':
        d = 0.5
    elif size == 'small':
        d = 0.1
    return w*d, h*d


def generate_union(position, distance, sub_w, sub_h, obj_w, obj_h):
    if distance == 'big':
        dw = (sub_w + obj_w) / 2 * 2
        dh = (sub_h + obj_h) / 2 * 2
    elif distance == 'mid':
        dw = (sub_w + obj_w) / 2 * 0.6
        dh = (sub_h + obj_h) / 2 * 0.6
    elif distance == 'small':
        dw = (sub_w + obj_w) / 2 * 0.1  
        dh = (sub_h + obj_h) / 2 * 0.1
    
    sub_x, sub_y = (-sub_w / 2, -sub_h / 2)
    if position == 'top':
        obj_x, obj_y = (-obj_w / 2, -obj_h / 2 - dh)
    elif position == 'bottom':
        obj_x, obj_y = (-obj_w / 2, -obj_h / 2 + dh)
    elif position == 'left':
        obj_x, obj_y = (-obj_w / 2 + dw, -obj_h / 2)
    elif position == 'right':
        obj_x, obj_y = (-obj_w / 2 - dw, -obj_h / 2)
    elif position == 'top_left':
        obj_x, obj_y = (-obj_w / 2 + dw, -obj_h / 2 -dh)
    elif position == 'top_right':
        obj_x, obj_y = (-obj_w / 2 - dw, -obj_h / 2 - dh)
    elif position == 'bottom_right':
        obj_x, obj_y = (-obj_w / 2 - dw, -obj_h / 2 + dh)
    elif position == 'bottom_left':
        obj_x, obj_y = (-obj_w / 2 + dw, -obj_h / 2 + dh)
   
    u_min_x, u_min_y = min(sub_x, obj_x), min(sub_y,obj_y)
    u_max_x, u_max_y = max(sub_x + sub_w, obj_x + obj_w), max(sub_y + sub_h, obj_y + obj_h)
    sub_x, sub_y, obj_x, obj_y = sub_x - u_min_x, sub_y - u_min_y, obj_x - u_min_x, obj_y - u_min_y 
    union_w = u_max_x - u_min_x
    union_h = u_max_y - u_min_y

    return sub_x, sub_y, obj_x, obj_y, union_w, union_h

def draw_union(spatial_img_path, sub_x, sub_y, sub_w, sub_h, obj_x, obj_y, obj_w, obj_h, union_w, union_h, name):
    fig_w=3
    fig_h=2
    fig,ax = plt.subplots(figsize=(fig_w,fig_h))
    norm_w = 1./ union_w
    norm_h = 1./ union_h
    tail_square = plt.Rectangle(xy=(obj_x*norm_w, obj_y*norm_h), width=obj_w*norm_w, height=obj_h*norm_h, linewidth=5,color='g', fill=True, alpha=0.5) 
    head_square = plt.Rectangle(xy=(sub_x*norm_w, sub_y*norm_h), width=sub_w*norm_w, height=sub_h*norm_h, linewidth=5, color='r', fill=True, alpha=0.5)

    ax.add_patch(tail_square)
    ax.add_patch(head_square)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(spatial_img_path, name + '.jpg'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw spatial imgs")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    sub_size_list = ['big', 'mid', 'small']
    sub_shape_list = ['horizontal', 'vertical', 'central']
    obj_size_list = ['big', 'mid', 'small']
    obj_shape_list = ['horizontal', 'vertical', 'central']
    distance_list = ['big', 'mid', 'small']
    position_list = ['top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right']


    for sub_size in sub_size_list:
        for sub_shape in sub_shape_list:
            sub_w, sub_h = generate_box(sub_size, sub_shape)
            for obj_size in obj_size_list:
                for obj_shape in obj_shape_list:
                    obj_w, obj_h = generate_box(obj_size, obj_shape)
                    for distance in distance_list:
                        for position in position_list:
                            sub_x, sub_y, obj_x, obj_y, union_w, union_h = generate_union(position, distance, sub_w, sub_h, obj_w, obj_h)
                            name = sub_size + '_' + sub_shape + '_' + obj_size + '_' + obj_shape + '_' + position + '_' + distance
                            draw_union(cfg.SPATIAL_IMG_PATH, sub_x, sub_y, sub_w, sub_h, obj_x, obj_y, obj_w, obj_h, union_w, union_h, name)

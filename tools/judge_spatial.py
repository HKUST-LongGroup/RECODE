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


# shape_list = ['horizontal', 'vertical', 'central']
def judge_bbox(w,h,union_area):
    area = w * h
    # print(area, union_area)
    if area <= 0.05 * union_area:
        size = 'small'
    elif area <= 0.5 * union_area:
        size = 'mid'
    else:
        size = 'big'
    if w <= h * 0.25:
        shape = 'vertical'
    elif h <= w * 0.25:
        shape = 'horizontal'
    else:
        shape = 'central'
    return size, shape

# distance_list = ['big', 'mid', 'small']
# position_list = ['top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
def judge_position(off_x, off_y, sub_w, sub_h, obj_w, obj_h):
    vec1 = np.array([off_x, off_y])
    vec2 = np.array([1, 0])
    cos_sim = vec1.dot(vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if cos_sim >= 0.8:
        position = 'left'
        if abs(off_x) > (sub_w + obj_w):
            distance = 'big'
        elif abs(off_x) < abs(sub_w / 2 - obj_w / 2):
            distance = 'small'
        else:
            distance = 'mid'
    elif cos_sim <= - 0.8:
        position = 'right'
        if abs(off_x) > (sub_w + obj_w):
            distance = 'big'
        elif abs(off_x) < abs(sub_w / 2 - obj_w / 2):
            distance = 'small'
        else:
            distance = 'mid'
    elif cos_sim <= 0.2 and cos_sim >= -0.2:
        if off_y < 0:
            position = 'bottom'
        else:
            position = 'top'
        if abs(off_y) > (sub_h + obj_h):
                distance = 'big'
        elif abs(off_y) < abs(sub_h / 2 - obj_h / 2):
            distance = 'small'
        else:
            distance = 'mid'
    elif cos_sim < 0.8 and cos_sim > 0.2:
        if off_y < 0:
            position = 'bottom_left'
        else:
            position = 'top_left'
        if abs(off_x) > (sub_w + obj_w) and abs(off_y) > (sub_h + obj_h) :
            distance = 'big'
        elif abs(off_x) < abs(sub_w/2 + obj_w/2) and abs(off_y) < abs(sub_h / 2 - obj_h / 2):
            distance = 'small'
        else:
            distance = 'mid'
    else:
        if off_y < 0:
            position = 'bottom_right'
        else:
            position = 'top_right'
        if abs(off_x) > (sub_w + obj_w) and abs(off_y) > (sub_h + obj_h) :
            distance = 'big'
        elif abs(off_x) < abs(sub_w/2 + obj_w/2) and abs(off_y) < abs(sub_h / 2 - obj_h / 2):
            distance = 'small'
        else:
            distance = 'mid'


    return position, distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge Spatial Info")
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

    clip_model_name = cfg.CLIP_MODEL_NAME.replace('/','_')
    box_proposals = torch.load(os.path.join(cfg.PRO_DIR,'{}_box_proposal_dict.pth'.format(clip_model_name)), map_location=torch.device('cpu'))

    idx2obj = {"1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}


    spatial_name_dict = {}

    for img_id in box_proposals.keys():

        proposal = box_proposals[img_id]
        num_prp = proposal.bbox.shape[0]
        n = len(proposal)
        cand_matrix = torch.ones((n, n)) - torch.eye(n)

        idxs = torch.nonzero(cand_matrix).view(-1,2)
        name_list = []
        if len(idxs) > 0:
            for idx in idxs:
                tgt_head_idx = idx[0]
                tgt_tail_idx = idx[1]
                xyxy_sub_bbox = proposal.bbox[tgt_head_idx]
                xyxy_obj_bbox = proposal.bbox[tgt_tail_idx]
                xywh_sub_bbox = proposal.convert('xywh').bbox[tgt_head_idx]
                xywh_obj_bbox = proposal.convert('xywh').bbox[tgt_tail_idx]
                union_w= max(xyxy_sub_bbox[2], xyxy_obj_bbox[2]) - min(xyxy_sub_bbox[0], xyxy_obj_bbox[0])
                union_h= max(xyxy_sub_bbox[3], xyxy_obj_bbox[3]) - min(xyxy_sub_bbox[1], xyxy_obj_bbox[1])
                union_x = min(xyxy_sub_bbox[0], xyxy_obj_bbox[0])
                union_y = min(xyxy_sub_bbox[1], xyxy_obj_bbox[1]) 
                union_area = union_w * union_h
                off_x, off_y = (xyxy_obj_bbox[2] + xyxy_obj_bbox[0]) / 2. - (xyxy_sub_bbox[2] + xyxy_sub_bbox[0]) / 2., (xyxy_obj_bbox[3] + xyxy_obj_bbox[1]) / 2. - (xyxy_sub_bbox[3] + xyxy_sub_bbox[1]) / 2.
                sub_x, sub_y, sub_w, sub_h = xywh_sub_bbox[0], xywh_sub_bbox[1], xywh_sub_bbox[2], xywh_sub_bbox[3]
                obj_x, obj_y, obj_w, obj_h = xywh_obj_bbox[0], xywh_obj_bbox[1], xywh_obj_bbox[2], xywh_obj_bbox[3]
                
                sub_size, sub_shape = judge_bbox(sub_w, sub_h, union_area)
                obj_size, obj_shape = judge_bbox(obj_w, obj_h, union_area)
                position, distance = judge_position(off_x, off_y, sub_w, sub_h, obj_w, obj_h)
                name = sub_size + '_' + sub_shape + '_' + obj_size + '_' + obj_shape + '_' + position + '_' + distance
                name_list.append(name)

        spatial_name_dict[img_id] = name_list
    torch.save(spatial_name_dict, os.path.join(cfg.PRO_DIR, '{}_spatial_name_test.pth'.format(clip_model_name)))


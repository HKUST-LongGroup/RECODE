import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from maskrcnn_benchmark.config import cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


sub_size_list = ['big', 'mid', 'small']
sub_shape_list = ['horizontal', 'vertical', 'central']
distance_list = ['big', 'mid', 'small']
position_list = ['top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
obj_size_list = ['big', 'mid', 'small']
obj_shape_list = ['horizontal', 'vertical', 'central']

def generate_spatial_logits(dataset, des_path, spatial_img_path, saved_path, clip_model_name):
   
    
    if dataset == 'VG':
        # VG
        id2pred = {"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
        SEMANTIC_GROUP = [11, 12, 13, 14, 15, 18, 19, 21, 24, 25, 26, 28, 34, 35, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47]
    # pred_spatial_prompts = dict(torch.load("/home/chenguikun/projects/Zero_Shot_SGG/exp/gpt3/3277_rel_sub_obj_prompts_chatgpt_new_pos_size.pth"))
   
    
    else:
        # GQA
        idx2pred = {"1": "on", "2": "wearing", "3": "of", "4": "near", "5": "in", "6": "behind", "7": "in front of", "8": "holding", "9": "next to", "10": "above", "11": "on top of", "12": "below", "13": "by", "14": "with", "15": "sitting on", "16": "on the side of", "17": "under", "18": "riding", "19": "standing on", "20": "beside", "21": "carrying", "22": "walking on", "23": "standing in", "24": "lying on", "25": "eating", "26": "covered by", "27": "looking at", "28": "hanging on", "29": "at", "30": "covering", "31": "on the front of", "32": "around", "33": "sitting in", "34": "parked on", "35": "watching", "36": "flying in", "37": "hanging from", "38": "using", "39": "sitting at", "40": "covered in", "41": "crossing", "42": "standing next to", "43": "playing with", "44": "walking in", "45": "on the back of", "46": "reflected in", "47": "flying", "48": "touching", "49": "surrounded by", "50": "covered with", "51": "standing by", "52": "driving on", "53": "leaning on", "54": "lying in", "55": "swinging", "56": "full of", "57": "talking on", "58": "walking down", "59": "throwing", "60": "surrounding", "61": "standing near", "62": "standing behind", "63": "hitting", "64": "printed on", "65": "filled with", "66": "catching", "67": "growing on", "68": "grazing on", "69": "mounted on", "70": "facing", "71": "leaning against", "72": "cutting", "73": "growing in", "74": "floating in", "75": "driving", "76": "beneath", "77": "contain", "78": "resting on", "79": "worn on", "80": "walking with", "81": "driving down", "82": "on the bottom of", "83": "playing on", "84": "playing in", "85": "feeding", "86": "standing in front of", "87": "waiting for", "88": "running on", "89": "close to", "90": "sitting next to", "91": "swimming in", "92": "talking to", "93": "grazing in", "94": "pulling", "95": "pulled by", "96": "reaching for", "97": "attached to", "98": "skiing on", "99": "parked along", "100": "hang on"}
        SEMANTIC_GROUP = [8, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 47]
    
    semantic_prompts = {}
    pred_spatial_prompts = dict(torch.load(des_path))
    for rel_sub_obj_key in pred_spatial_prompts.keys():
        prompts_list = pred_spatial_prompts[rel_sub_obj_key]['pos']
        sub_cat = rel_sub_obj_key.split('_')[1]
        obj_cat = rel_sub_obj_key.split('_')[2]
        filter_idxs = []
        # print(prompts_list)
        for idx, prompt in enumerate(prompts_list):
            if len(prompt) >1:
                filter_idxs.append(idx)
        prompts_list = np.array(prompts_list)[filter_idxs]
        new_prompts_list = []
        for idx in range(len(prompts_list)):
            prompt = prompts_list[idx].item()
            prompt = prompt.strip()
            prompt = prompt.lower()
            prompt = prompt.replace(' \n', '.')
            prompt = prompt.replace('\n', '.')
            prompt = prompt.replace('.', '')
            if 'subject' in prompt:
                prompt = prompt.replace('subject', 'red box')
            if 'object' in prompt:
                prompt = prompt.replace('object', 'green box')
            prompt = prompt.replace(sub_cat, 'red box')
            prompt = prompt.replace(obj_cat, 'green box')
            prompt = prompt.capitalize()
            new_prompts_list.append(prompt)
        semantic_prompts[rel_sub_obj_key] = clip.tokenize(new_prompts_list).to(device)


    spatial_logis_dict = {}
    semantic_logits = {}
    for sub_size in sub_size_list:
        for sub_shape in sub_shape_list:
            for obj_size in obj_size_list:
                for obj_shape in obj_shape_list:
                    for distance in distance_list:
                        for position in position_list:
                            name = sub_size + '_' + sub_shape + '_' + obj_size + '_' + obj_shape + '_' + position + '_' + distance
                            img = Image.open(os.path.join(spatial_img_path, name + ".jpg"))
                            image = preprocess(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                image_feature = model.encode_image(image)
                                for rel_sub_obj_key in pred_spatial_prompts.keys():
                                    text = semantic_prompts[rel_sub_obj_key]
                                    if len(text) > 0:
                                        text_feature = model.encode_text(text)
                                        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                                        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                                        semantic_logits[rel_sub_obj_key] = (100.0 * image_feature @ text_feature.T).float().mean()
                                spatial_logis_dict[name] = semantic_logits  
    torch.save(spatial_logis_dict,  os.path.join(saved_path, '{}_spatial_logits.pth'.format(clip_model_name)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spatial logits")
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
    generate_spatial_logits(cfg.SELECT_DATASET, cfg.GPT_DES_PATH, cfg.SPATIAL_IMG_PATH, cfg.GPT_DIR, clip_model_name)
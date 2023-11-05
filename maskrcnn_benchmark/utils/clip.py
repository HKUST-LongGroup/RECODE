from turtle import pd
import clip
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
import torch.nn.functional as F
import os

class ClipPredictor(nn.Module):
    
    def __init__(self, device, model_name='ViT-B/32'):
        super(ClipPredictor, self).__init__()
        self.device = device
        self.model_name = cfg.CLIP_MODEL_NAME
        self.model, self.preprocess = clip.load(self.model_name, device=device)
        self.model.eval()
        self.img_mean = np.array(cfg.INPUT.PIXEL_MEAN)
        self.img_std = np.array(cfg.INPUT.PIXEL_STD)
        self.idx2prdc, self.idx2obj = self._get_dict_info()
        self.obj_text_embeds = self._get_obj_text_embeds()
        self.reasonable_tris = None
        self.num_rels = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.pair_text_embeds = {}
        self.sub_rel_text_features_dict_gpt = {}
        self.obj_rel_text_features_dict_gpt = {}
        self.sub_rel_text_features_dict = {}
        self.obj_rel_text_features_dict = {}
        self.pair_texts = {}
        self.weights_dict = np.load(cfg.WEIGHT_PATH, allow_pickle=True).item()
        self.spatial_logits_dict = torch.load(cfg.GPT_POS_PATH, map_location=device)
        self.chatgpt_dict = torch.load(cfg.GPT_DES_PATH, map_location=device)
        self.valid_sub_pred = np.load(cfg.SUB_PATH, allow_pickle=True).item()
        self.valid_obj_pred = np.load(cfg.OBJ_PATH, allow_pickle=True).item()
        if cfg.SELECT_DATASET == 'GQA':
            self.semantic_group = cfg.GQA_SEMANTIC_GROUP
        else:
            self.semantic_group = cfg.SEMANTIC_GROUP
            
    def _get_dict_info(self):
        dataset_args = DatasetCatalog.get(cfg.DATASETS.TEST[0], cfg)
        info = json.load(open(dataset_args['args']['dict_file'], 'r'))
        idx2prdc = {int(k): v for k, v in info['idx_to_predicate'].items()}
        idx2obj = {int(k): v for k, v in info['idx_to_label'].items()}
        return idx2prdc, idx2obj

    def _get_obj_text_embeds(self):
        prompts = [self._get_cls_prompt(self.idx2obj[idx]) for idx in range(1, len(self.idx2obj) + 1, 1)]
        obj_text_embeds = clip.tokenize(prompts).to(self.device)
        return obj_text_embeds

    def _get_cls_prompt(self, cls_name):
        assert len(cls_name) > 1
        return f'A photo of an {cls_name}.' if cls_name[0] in 'aeiou' else f'A photo of a {cls_name}.'
    
    def _get_rel_prompt(self, pairs):
        """
        name of subject _ name of object
        """
        sub, obj = pairs.split('_')
        prompts = ['There is no visual relation between \'{}\' and \'{}\'.'.format(sub, obj)]
        prompts.extend(['A visual triplet \'{}, {}, {}\'.'.format(sub, self.idx2prdc[idx], obj) for idx in range(1, len(self.idx2prdc) + 1, 1)])
        return prompts, clip.tokenize(prompts).to(self.device)

    def _inv_img_tensors(self, img):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * self.img_std + self.img_mean)[:, :, ::-1].astype(np.uint8)
        return img
    
    def _clamp_bbox(self, bboxes, shape):
        bboxes = torch.clamp(bboxes, min=0)
        bboxes[:, 2] = torch.clamp(bboxes[:, 2], max=shape[1])
        bboxes[:, 3] = torch.clamp(bboxes[:, 3], max=shape[0])
        bboxes = bboxes.tolist()
        return bboxes

    def _get_obj_dist(self, img, proposal):
        img = self._inv_img_tensors(img)
        bboxes = self._clamp_bbox(proposal.bbox.long(), img.shape)
        regions = []
        for bbox in bboxes:
            region = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            region = self.preprocess(Image.fromarray(region).convert('RGB'))
            regions.append(region.unsqueeze(0).to(self.device))
        regions = torch.cat(regions, dim=0)
        with torch.no_grad():
            logits_per_image, _ = self.model(regions, self.obj_text_embeds)
            probs = logits_per_image.softmax(dim=-1)
            zero_vec = torch.zeros((probs.shape[0], 1), dtype=probs.dtype, device=self.device)
            probs = torch.cat([zero_vec, probs], dim=-1)
        return probs
    
    def _get_obj_dist_offline(self, obj_features):
        with torch.no_grad():
            text_features = self.model.model.encode_text(self.obj_text_embeds)
            obj_features /= obj_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * obj_features @ text_features.T)
        return similarity

    
    def _get_obj_feature(self, img, proposal):
        img = self._inv_img_tensors(img)
        bboxes = self._clamp_bbox(proposal.bbox.long(), img.shape)
        regions = []
        for bbox in bboxes:
            region = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            region = self.preprocess(Image.fromarray(region))
            regions.append(region.unsqueeze(0).to(self.device))
        regions = torch.cat(regions, dim=0)
        with torch.no_grad():
            clip_obj_features = self.model.encode_image(regions)
        return clip_obj_features
    
    def _get_union_info(self, proposal, rel_pair_idx):
        predict_logits = proposal.get_field('predict_logits')
        predict_labels = torch.argmax(predict_logits, dim=-1)
        sub_labels = predict_labels[rel_pair_idx[:, 0]].tolist()
        obj_labels = predict_labels[rel_pair_idx[:, 1]].tolist()
        head_proposal = proposal[rel_pair_idx[:, 0]]
        tail_proposal = proposal[rel_pair_idx[:, 1]]
        union_proposal = boxlist_union(head_proposal, tail_proposal)
        name_pairs = ['{}_{}'.format(self.idx2obj[sub_labels[i]], self.idx2obj[obj_labels[i]]) for i in range(len(sub_labels))]
        pair_embeds = []
        pair_prompts = []
        for pairs in name_pairs:
            if pairs not in self.pair_text_embeds:
                prompts, embed = self._get_rel_prompt(pairs)
                self.pair_text_embeds[pairs] = embed
                self.pair_texts[pairs] = prompts
            pair_embeds.append(self.pair_text_embeds[pairs])
            pair_prompts.append(self.pair_texts[pairs])
        return union_proposal, pair_embeds, pair_prompts

    

    def _get_rel_dist(self, img, proposal, rel_pair_idx):
        img = self._inv_img_tensors(img)
        union_proposal, rel_text_embeds, rel_texts = self._get_union_info(proposal, rel_pair_idx)
        union_bboxes = self._clamp_bbox(union_proposal.bbox.long(), img.shape)
        regions = []
        for bbox in union_bboxes:
            region = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            region = self.preprocess(Image.fromarray(region).convert('RGB'))
            regions.append(region.unsqueeze(0).to(self.device))
        assert len(regions) == len(rel_text_embeds)
        with torch.no_grad():
            probs = []
            prompts = []
            for region, embed, prompt in zip(regions, rel_text_embeds, rel_texts):
                logits_per_image, _ = self.model(region, embed)
                probs.append(logits_per_image)
                prompts.append(prompt)
            probs = torch.cat(probs, dim=0)
        return probs, prompts

    def _get_sub_rel_text_embeds(self, sub):
        if cfg.CP == True:
            prompts = ['{} has no relation.'.format(sub)]
            prompts.extend(['{} {} something.'.format(sub, self.idx2prdc[idx]) for idx in range(1, len(self.idx2prdc) + 1, 1)])
        else: 
            if cfg.PHOTO == False:
                prompts = ['no relation.']
                prompts.extend(['{}'.format(self.idx2prdc[idx]) for idx in range(1, len(self.idx2prdc) + 1, 1)])
            else:
                prompts = ['no relation.']
                prompts.extend(['a photo of {}.'.format(self.idx2prdc[idx]) for idx in range(1, len(self.idx2prdc) + 1, 1)])
        return clip.tokenize(prompts).to(self.device)

    def _get_obj_rel_text_embeds(self, obj):
        if cfg.CP ==  True:
            prompts = ['{} has no relation.'.format(obj)]
            prompts.extend(['something {} {}.'.format(self.idx2prdc[idx], obj) for idx in range(1, len(self.idx2prdc) + 1, 1)])
        else:
            if cfg.PHOTO == False:
                prompts = ['no relation.']
                prompts.extend(['{}'.format(self.idx2prdc[idx]) for idx in range(1, len(self.idx2prdc) + 1, 1)])
            else:
                prompts = ['no relation.']
                prompts.extend(['a photo of {}.'.format(self.idx2prdc[idx]) for idx in range(1, len(self.idx2prdc) + 1, 1)])
        return clip.tokenize(prompts).to(self.device)
    
    def _get_rel_dist_offline(self, entity_features, predict_labels, rel_pair_idx):

        sub_features = entity_features[rel_pair_idx[:, 0]]
        obj_features = entity_features[rel_pair_idx[:, 1]]

        sub_labels = predict_labels[rel_pair_idx[:, 0]].tolist()
        obj_labels = predict_labels[rel_pair_idx[:, 1]].tolist()
        sub_rel_text_features = []
        obj_rel_text_features = []
        for sub_label in sub_labels:
            sub_label = self.idx2obj[sub_label]
            if sub_label not in self.sub_rel_text_features_dict:
                sub_prompts = self._get_sub_rel_text_embeds(sub_label)
                with torch.no_grad():
                    sub_text_feature = self.model.encode_text(sub_prompts)
                self.sub_rel_text_features_dict[sub_label] = sub_text_feature
            sub_rel_text_features.append(self.sub_rel_text_features_dict[sub_label])
        
        for obj_label in obj_labels:
            obj_label = self.idx2obj[obj_label]
            if obj_label not in self.obj_rel_text_features_dict:
                obj_prompts = self._get_obj_rel_text_embeds(obj_label)
                with torch.no_grad():
                    obj_text_feature = self.model.encode_text(obj_prompts)
                self.obj_rel_text_features_dict[obj_label] = obj_text_feature
            obj_rel_text_features.append(self.obj_rel_text_features_dict[obj_label])

        sub_similarities = []
        for sub_feature, sub_text_feature in zip(sub_features, sub_rel_text_features):
            sub_feature = sub_feature / sub_feature.norm(dim=-1, keepdim=True)
            sub_text_feature = sub_text_feature / sub_text_feature.norm(dim=-1, keepdim=True)
            sub_similarity = (100.0 * sub_feature @ sub_text_feature.T)
            if cfg.TEST_GROUP == 'semantic':
                filter_idxs = np.array([int(idx) for idx in range(self.num_rels) if idx not in self.semantic_group])
                sub_similarity[filter_idxs] = 0
            sub_similarities.append(sub_similarity.view(1,-1))

        obj_similarities = []
        for obj_feature, obj_text_feature in zip(obj_features, obj_rel_text_features):
            obj_feature = obj_feature / obj_feature.norm(dim=-1, keepdim=True)
            obj_text_feature = obj_text_feature / obj_text_feature.norm(dim=-1, keepdim=True)
            obj_similarity = (100.0 * obj_feature @ obj_text_feature.T)
            if cfg.TEST_GROUP == 'semantic':
                filter_idxs = np.array([int(idx) for idx in range(self.num_rels) if idx not in self.semantic_group])
                obj_similarity[filter_idxs] = 0
            obj_similarities.append(obj_similarity.view(1,-1))
        
        for i ,(sub_label_idx, obj_label_idx) in enumerate(zip(sub_labels, obj_labels)):
            for idx in self.semantic_group:
                pred_label = self.idx2prdc[idx]
                sub_label = self.idx2obj[sub_label_idx]
                obj_label = self.idx2obj[obj_label_idx]
                tri = pred_label +'_'+ sub_label +'_'+ obj_label
                if cfg.FILTER_CAT == True:
                    if cfg.PAIR == True:
                        sub_pred_label = sub_label + '_' + pred_label
                        obj_pred_label = obj_label + '_' + pred_label
                        if bool(self.valid_sub_pred[sub_pred_label]) == False or bool(self.valid_obj_pred[obj_pred_label]) == False:
                            sub_similarities[i][:,idx]=0
                            obj_similarities[i][:,idx]=0

        probs = torch.cat(sub_similarities, dim=0) + torch.cat(obj_similarities, dim=0)

        return probs

    def _judge_cat(self, label):
        if label in cfg.HUMAN:
            return 'human'
        elif label in cfg.ANIMAL:
            return 'animal'
        else:
            return 'product'
        
    def _get_rel_cat_sub_gpt3_per_rel(self, sub_label, obj_label, pred_label):
        sub_cat = self._judge_cat(sub_label)
        obj_cat = self._judge_cat(obj_label)
        pred_sub_obj_key = pred_label + '_' + sub_cat + '_' + obj_cat
        sub_rel_gpt3_prompts_dict = self.chatgpt_dict
        if pred_sub_obj_key in sub_rel_gpt3_prompts_dict.keys():
            sub_rel_gpt3_prompts = sub_rel_gpt3_prompts_dict[pred_sub_obj_key]
            if 'sub' in sub_rel_gpt3_prompts.keys():
                prompts_list = sub_rel_gpt3_prompts['sub']
                filter_idxs = []
                new_prompts_list = []
                for idx, prompt in enumerate(prompts_list):
                    if len(prompt) >1:
                        filter_idxs.append(idx)
                for idx in filter_idxs:
                    prompt = prompts_list[idx].item()
                    prompt = sub_label + ', ' + prompt
                    prompt = prompt.strip()
                    prompt = prompt.replace(' \n', '.')
                    prompt = prompt.replace('\n', '.')
                    prompt = prompt.replace('.', '')
                    prompt = prompt.replace('subject', sub_label)
                    prompt = prompt.replace('the object', 'something')
                    prompt = prompt.replace('object', 'something')
                    new_prompts_list.append(prompt)
            else:
                new_prompts_list = ['{} {} something'.format(sub_label, pred_label)]
            return clip.tokenize(new_prompts_list).to(self.device), new_prompts_list
        else:
            return None, None

    def _get_rel_cat_obj_gpt3_per_rel(self, sub_label, obj_label, pred_label):

        sub_cat = self._judge_cat(sub_label)
        obj_cat = self._judge_cat(obj_label)
        pred_sub_obj_key = pred_label + '_' + sub_cat + '_' + obj_cat
        obj_rel_gpt3_prompts_dict = self.chatgpt_dict
        if pred_sub_obj_key in obj_rel_gpt3_prompts_dict.keys():
            obj_rel_gpt3_prompts = obj_rel_gpt3_prompts_dict[pred_sub_obj_key]
            if 'obj' in obj_rel_gpt3_prompts.keys():
                prompts_list = obj_rel_gpt3_prompts['obj']
                filter_idxs = []
                new_prompts_list = []
                for idx, prompt in enumerate(prompts_list):
                    if len(prompt) >1:
                        filter_idxs.append(idx)
                for idx in filter_idxs:
                    prompt = prompts_list[idx].item()
                    prompt = obj_label + ', ' + prompt
                    prompt = prompt.strip()
                    prompt = prompt.replace(' \n', '.')
                    prompt = prompt.replace('\n', '.')
                    prompt = prompt.replace('.', '')
                    prompt = prompt.replace('object', obj_label)
                    prompt = prompt.replace('the subject', 'something')
                    prompt = prompt.replace('subject', 'something')
                    new_prompts_list.append(prompt)
            else:
                new_prompts_list = ['something {} {}'.format(pred_label, obj_label)]
            return clip.tokenize(new_prompts_list).to(self.device), new_prompts_list
        else:
            return None, None
    
    def _get_rel_dist_gpt_des(self, entity_features, predict_labels, rel_pair_idx, topk, rel_labels, spatial_names):
        device = torch.device('cuda')
        weights_dict = self.weights_dict
        spatial_logits_dict = self.spatial_logits_dict
        sub_features = entity_features[rel_pair_idx[:, 0]]
        obj_features = entity_features[rel_pair_idx[:, 1]]
        sub_labels = predict_labels[rel_pair_idx[:, 0]].tolist()
        obj_labels = predict_labels[rel_pair_idx[:, 1]].tolist()
        device = entity_features[0].device
        probs = torch.zeros(len(rel_pair_idx), self.num_rels).to(device)

        for i, (sub_label, obj_label, sub_feature, obj_feature, spatial_name) in enumerate(zip(sub_labels, obj_labels, sub_features, obj_features, spatial_names)):

            sub_label = self.idx2obj[sub_label]
            obj_label = self.idx2obj[obj_label]
            sub_prob = torch.zeros(self.num_rels).to(device)
            obj_prob = torch.zeros(self.num_rels).to(device)
            pos_prob = torch.zeros(self.num_rels).to(device)

            sub_rel_text_features = []
            obj_rel_text_features = []

            for pred_label_id in self.semantic_group:
                pred_label = self.idx2prdc[pred_label_id]
                sub_cat = self._judge_cat(sub_label)
                obj_cat = self._judge_cat(obj_label)
                tri = pred_label + '_' + sub_label + '_' + obj_label
                pred_sub_obj_key = pred_label + '_' + sub_cat + '_' + obj_cat
                sub_key = sub_label + '_' + pred_label + '_' + obj_cat
                obj_key = sub_cat + '_' + pred_label + '_' + obj_label
                if cfg.PAIR == True:
                    sub_pred_label = sub_label + '_' + pred_label
                    obj_pred_label = obj_label + '_' + pred_label
                if (cfg.FILTER_CAT==False and pred_sub_obj_key in weights_dict.keys()) or (cfg.FILTER_CAT==True and cfg.PAIR==True and pred_sub_obj_key in weights_dict.keys() and bool(self.valid_sub_pred[sub_pred_label]) == True and bool(self.valid_obj_pred[obj_pred_label]) == True) or (cfg.FILTER_CAT==True and cfg.PAIR==False and pred_sub_obj_key in weights_dict.keys() and tri in self.reasonable_tris):
                    if sub_key not in self.sub_rel_text_features_dict_gpt:
                        sub_prompts_embeds, sub_prompts = self._get_rel_cat_sub_gpt3_per_rel(sub_label, obj_label, pred_label)
                        if sub_prompts_embeds != None:
                            with torch.no_grad():
                                sub_text_feature = self.model.encode_text(sub_prompts_embeds)
                        else:
                            sub_text_feature = None
                        self.sub_rel_text_features_dict_gpt[sub_key] = sub_text_feature
                    sub_rel_text_features.append(self.sub_rel_text_features_dict_gpt[sub_key])
            
                    if obj_key not in self.obj_rel_text_features_dict_gpt:
                        obj_prompts_embeds, obj_prompts = self._get_rel_cat_obj_gpt3_per_rel(sub_label, obj_label, pred_label)
                        if obj_prompts_embeds != None:
                            with torch.no_grad():
                                obj_text_feature = self.model.encode_text(obj_prompts_embeds)
                        else:
                            obj_text_feature = None
                        self.obj_rel_text_features_dict_gpt[obj_key] = obj_text_feature
                    obj_rel_text_features.append(self.obj_rel_text_features_dict_gpt[obj_key])
                else:
                    sub_rel_text_features.append(None)
                    obj_rel_text_features.append(None)

            for idx in range(len(self.semantic_group)):
                sub_cat = self._judge_cat(sub_label)
                obj_cat = self._judge_cat(obj_label)
                pred_label = self.idx2prdc[self.semantic_group[idx]]
                pred_sub_obj_key = pred_label + '_' + sub_cat + '_' + obj_cat
                tri = pred_label + '_' + sub_label + '_' + obj_label
                if cfg.PAIR == True:
                    sub_pred_label = sub_label + '_' + pred_label
                    obj_pred_label = obj_label + '_' + pred_label
                if (cfg.FILTER_CAT==False and pred_sub_obj_key in weights_dict.keys()) or (cfg.FILTER_CAT==True and cfg.PAIR==True and pred_sub_obj_key in weights_dict.keys() and bool(self.valid_sub_pred[sub_pred_label]) == True and bool(self.valid_obj_pred[obj_pred_label]) == True) or (cfg.FILTER_CAT==True and cfg.PAIR==False and pred_sub_obj_key in weights_dict.keys() and tri in self.reasonable_tris):

                    if cfg.GPT_WEIGHT:
                        weight = weights_dict[pred_sub_obj_key]
                        if cfg.SPATIAL == True:
                            weight[0] = weight[0]/(sum(weight) + 1e-3)
                            weight[1] = weight[1]/(sum(weight) + 1e-3)
                            weight[2] = weight[2]/(sum(weight) + 1e-3)
                        else:
                            weight[2] = 0
                            weight[0] = weight[0]/(sum(weight) + 1e-3)
                            weight[1] = weight[1]/(sum(weight) + 1e-3)

                    else:
                        if cfg.SPATIAL:
                            weight = np.array([0.33, 0.33, 0.33])  
                        else:
                            weight = np.array([0.5, 0.5, 0.0])
                    weight = weight * cfg.WEIGHT_SUM
                    if pred_sub_obj_key in spatial_logits_dict[spatial_name].keys():
                        pos_prob[self.semantic_group[idx]] = spatial_logits_dict[spatial_name][pred_sub_obj_key]

                    sub_text_feature = sub_rel_text_features[idx]
                    if sub_text_feature == None or len(sub_text_feature) <=0 :
                        pass
                    else:
                        sub_feature = sub_feature / sub_feature.norm(dim=-1, keepdim=True)
                        sub_text_feature = sub_text_feature / sub_text_feature.norm(dim=-1, keepdim=True)
                        sub_prob[self.semantic_group[idx]] = (100.0 * sub_feature @ sub_text_feature.T).float().mean()

                    obj_text_feature = obj_rel_text_features[idx]
                    if obj_text_feature == None or len (obj_text_feature) <=0:
                        pass
                    else:
                        obj_feature = obj_feature / obj_feature.norm(dim=-1, keepdim=True)
                        obj_text_feature = obj_text_feature / obj_text_feature.norm(dim=-1, keepdim=True)
                        obj_prob[self.semantic_group[idx]] = (100.0 * obj_feature @ obj_text_feature.T).float().mean()
                    probs[i][self.semantic_group[idx]] = weight[0] * sub_prob[self.semantic_group[idx]] + weight[1] * obj_prob[self.semantic_group[idx]] + weight[2] * pos_prob[self.semantic_group[idx]] * 0.6
        pred_logits = probs
        return pred_logits
    
    # using union feature
    @torch.no_grad()
    def forward(self, images, proposals, rel_labels=None, rel_pair_idxs=None, infer_type='obj_cls', feat_type='offline', obj_labels=None, spatials=None):
        if rel_pair_idxs is not None:
            assert infer_type == 'rel_cls'
        if infer_type == 'obj_cls':
            class_logits = []
            if feat_type == 'offline':
                for img_idx in range(len(images)):
                    class_logits.append(self._get_obj_dist_offline(images[img_idx]))
                    pred_obj_labels = [class_logit.softmax(dim=-1)[:, 1:].max(dim=1) + 1 for class_logit in class_logits]
            else:
                for img_idx in range(len(proposals)):
                    class_logits.append(self._get_obj_dist(images.tensors[img_idx], proposals[img_idx]))
                    pred_obj_labels = [class_logit.softmax(dim=-1)[:, 1:].max(dim=1) + 1 for class_logit in class_logits]
            class_logits = torch.cat(class_logits, dim=0)
            return class_logits, pred_obj_labels
        elif infer_type == 'rel_cls':
            rel_logots = []
            rel_prompts_list = []
            if feat_type == 'offline':
                for img_idx in range(len(images)):
                    if len(torch.nonzero(rel_labels[img_idx])) <=0:
                        probs = torch.zeros(len(rel_pair_idxs[img_idx]), self.num_rels).to(self.device)
                    else:
                        if cfg.BOTH == True:
                            probs_clip = self._get_rel_dist_offline(images[img_idx], obj_labels[img_idx], rel_pair_idxs[img_idx])
                            probs_gpt = self._get_rel_dist_gpt_des(images[img_idx], obj_labels[img_idx], rel_pair_idxs[img_idx], topk=cfg.TOPK, rel_labels=rel_labels[img_idx], spatial_names=spatials[img_idx])
                            probs = (probs_clip + probs_gpt)/cfg.TAU
                        elif cfg.GPT == False:
                            probs = self._get_rel_dist_offline(images[img_idx], obj_labels[img_idx], rel_pair_idxs[img_idx])
                        elif cfg.GPT == True:
                            probs = self._get_rel_dist_gpt_des(images[img_idx], obj_labels[img_idx], rel_pair_idxs[img_idx], topk=cfg.TOPK, rel_labels=rel_labels[img_idx], spatial_names=spatials[img_idx])
                    rel_logots.append(probs)
            else:
                for img_idx in range(len(proposals)):
                    probs, rel_prompts = self._get_rel_dist(images.tensors[img_idx], proposals[img_idx], rel_pair_idxs[img_idx])
                    rel_logots.append(probs)
                    rel_prompts_list.append(rel_prompts)
                for i in range(len(rel_labels[0])):
                    if rel_labels[0][i] > 0:
                        rel_class_prob = F.softmax(rel_logots[0][i], -1)
                        rel_class = rel_class_prob[1:].argmax()
                        print(self.idx2prdc[rel_class.item() + 1], ',', self.idx2prdc[rel_labels[0][i].item()], ',', rel_prompts_list[0][i][rel_class.item() + 1])

            return rel_logots

    @torch.no_grad()
    def forward_extract_feature(self, images, proposals):
        clip_obj_features = []
        for img_idx in range(len(proposals)):
            clip_obj_features.append(self._get_obj_feature(images.tensors[img_idx], proposals[img_idx]))
        return clip_obj_features
    
clip_predictor = ClipPredictor('cuda')

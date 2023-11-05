CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10066 --nproc_per_node=1 tools/relation_test_net_clip.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_clip.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR ClipPredictor SOLVER.IMS_PER_BATCH 512 TEST.IMS_PER_BATCH 512 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/recode DATASETS.TO_TEST 'test' FEAT_DIR ./exp/vg/clip_obj_feature PRO_DIR ./exp/vg/proposal TEST_GROUP semantic TEST.ALLOW_LOAD_FROM_CACHE False BOTH True GPT_DIR  ./exp/vg GPT False GPT_WEIGHT True FILTER_CAT True WEIGHT_SUM 2.0 SPATIAL True PAIR True TAU 1.0 WEIGHT_PATH ./exp/vg/des_weight.npy SUB_PATH ./exp/vg/sub_valid_tris.npy OBJ_PATH ./exp/vg/obj_valid_tris.npy GPT_DES_PATH ./exp/vg/des_prompts.pth PHOTO False GPT_POS_PATH ./exp/vg/ViT-B_32_spatial_logits.pth
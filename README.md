# CFA for SGG in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.13.o-%237732a8)

Our paper [Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models](https://arxiv.org/abs/2305.12476) has been accepted by NIPS 2023.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Extract CLIP Visual Features
```base
bash scripts/extract_clip_obj_feature.sh
```

## Generate Spatial Images and Offline Spatial Logits 

```base
bash scripts/draw_imgs_and_generate_spatial_logits.sh
```


## Inference with RECODE

```base
bash scripts/infer.sh
```
## Generated Files
We provide the extracted clip visual feature, visual cue descriptions, and some spatial information, you can download from [here*](https://mega.nz/folder/BFxGQIxL#H-aKabcI-FnBlLlEEpR0uQ).

## Citations

If you find this project helps your research, please kindly consider citing our paper in your publications.

```
@article{li2023zero,
  title={Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models},
  author={Li, Lin and Xiao, Jun and Chen, Guikun and Shao, Jian and Zhuang, Yueting and Chen, Long},
  journal={arXiv preprint arXiv:2305.12476},
  year={2023}
}
```
## Credits

Our codebase is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
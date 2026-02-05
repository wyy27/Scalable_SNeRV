# SNeRV

This repository accompanies the paper "**SNeRV: Scalable Neural Representations for Video Coding**" (Workshop on Machine Learning and Compression @ NeurIPS 2024).

[Paper](https://openreview.net/forum?id=ZqN4bnXSSY&referrer=%5Bthe%20profile%20of%20Yiying%20Wei%5D(%2Fprofile%3Fid%3D~Yiying_Wei1))   |   [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/98211.png?t=1733609293.4857748)

## Quick Start
- Install dependencies:

```
pip install -r requirements.txt
```

## Environment
- PyTorch: 2.1.0 (see [requirements.txt](requirements.txt))
- Optional: [FFmpeg](https://www.ffmpeg.org/) for dataset preparation

## Dataset Preparation (UVG)
The UVG dataset can be downloaded from the [official page](https://ultravideo.fi/dataset.html).

Downscale to lower resolutions (270p, 540p) using FFmpeg, e.g.:

```
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -r 120 \
  -i Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv \
  -vf scale=480:270 \
  -f rawvideo -pix_fmt yuv420p \
  Bosphorus_480x270_120fps_420_8bit_YUV.yuv

ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -r 120 \
  -i Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv \
  -vf scale=960:540 \
  -f rawvideo -pix_fmt yuv420p \
  Bosphorus_960x540_120fps_420_8bit_YUV.yuv
```

Convert videos into PNG frames (example for 1080p Bosphorus):

```
mkdir Bosphorus
ffmpeg -video_size 1920x1080 -pixel_format yuv420p \
  -i Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv Bosphorus/%4d.png
```

Recommended directory structure:

```
Datasets/UVG/
├── 480x270/
│     ├── Beauty
│     │       ├── 0001.png
│     │       ├── 0002.png
│     │       └── ...
│     ├── Bosphorus
│     └── ...
├── 960x540/
│       └── ...
└── 1920x1080/
```

## Training (Encoding)
```
dataset_dir=~/Datasets/UVG/1920x1080
dataset_name=Bosphorus
output=~/Models/UVG_3layer_60fps/
train_cfg=$(cat "cfgs/train/snerv.txt")
model_cfg=$(cat "cfgs/models/uvg-snerv-l.txt")
accelerate launch --main_process_port 29631 --mixed_precision=fp16 --dynamo_backend=inductor snerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 30 --grad-accum 1 --seed 0
```

## Inference (Decoding)
```
dataset_dir=~/Datasets/UVG/1920x1080
dataset_name=Bosphorus
output=~/Models/UVG_3layer_60fps/
train_cfg=$(cat "cfgs/train/snerv.txt")
model_cfg=$(cat "cfgs/models/uvg-snerv-s.txt")
checkpoint_path=~/Models/UVG_3layer_60fps/Bosphorus-SNeRV-20260126-124211-fca8d171
accelerate launch --mixed_precision=fp16 --dynamo_backend=inductor snerv_main.py \
  --dataset ${dataset_dir} --dataset-name ${dataset_name} --output ${output} \
  ${train_cfg} ${model_cfg} --batch-size 144 --eval-batch-size 1 --grad-accum 1 --log-eval true --seed 0 \
  --bitstream ${checkpoint_path} --bitstream-q 6 --eval-only
```

Notes:
- PSNR/VMAF in the paper are computed in YUV420; RGB metrics tend to be higher. 

## Acknowledgement
This implementation builds upon code from [HiNeRV](https://github.com/hmkx/HiNeRV), [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) and [HNeRV](https://github.com/haochen-rye/HNeRV). We sincerely thank the contributors for their excellent work.

## Citation
Please consider citing our work if you find it useful.

```
@inproceedings{
wei2024snerv,
title={{SNeRV: Scalable Neural Representations for Video Coding}},
author={Yiying Wei and Hadi Amirpour and Christian Timmerer},
booktitle={Workshop on Machine Learning and Compression, NeurIPS 2024},
year={2024},
url={https://openreview.net/forum?id=ZqN4bnXSSY}
}

@inproceedings{wei2025nsvc,
  title={{Neural Representations for Scalable Video Coding}}, 
  author={Wei, Yiying and Amirpour, Hadi and Timmerer, Christian},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/ICME59968.2025.11209859}}
```

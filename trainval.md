## 1. Prepare the Pretrained Weights
Although some weights can be downloaded dynamically at runtime, it is recommended to pre-download them for speeding up each run.

#### Pre-trained Image Encoder (EVA ViT-g)
```
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```
the path of image encoder weight can be modified [here](stllm/models/eva_vit.py#L433).

#### Pre-trained Q-Former and Linear Projection
```
# InstructBLIP (recommended)
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
```
```
# MiniGPT4
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
wget https://huggingface.co/Vision-CAIR/MiniGPT-4/blob/main/pretrained_minigpt4.pth
```
the path of Q-Former and Linear Weight can be modified in ```q_former_model``` and ```ckpk``` in each config [here](config).

#### Prepare Vicuna Weights
Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.1 (for InstructBLIP) or Vicuna v1.0 (for MiniGPT4). 
Then modify the ```llama_model``` in each config [here](config) to the folder that contains Vicuna weights.

## 2. Training 
#### Data
We follow [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) to maintain consistency in the format of each instruction dataset. 
Please follow the source [instructions](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/DATA.md) to prepare the videos and annotations for each dataset.
Then modify the path for each dataset [here](stllm/datasets/datasets/instruction_data.py).

Please note：

(1）We do not need to prepare all datasets; we only need to prepare the datasets corresponding to the configurations needed for execution.

(2) The annotations for videochat11k and videochatgpt100k are slightly different from the source, which can be found [here](https://drive.google.com/file/d/1HIcT0WOmnHNU_xLtezKaHeUG8qa0_wQh/view).

#### Running
Please first modify the path in [train script](script/train/train.sh) for the desired config from [config folder](config), then run
```
bash script/train/train.sh
```

## 3. Inference
#### MVBench 
Please first modify the checkpoint path and annotation path in [test script], then run 
```
bash script/inference/mvbench/test_mvbench.sh
```

#### VcgBench 
All evaluation scripts can be found [here](script/inference/vcgbench).

For instance, to evaluate the temporal score on VideoChatGPT benchmark, we first run the inference to get prediction results: 
```
bash script/inference/vcgbench/test_temporal.sh
```
and then execute the corresponding evaluation script to perform benchmarking:
```
bash script/inference/vcgbench/score_temporal.sh
```

#### VideoQABench
All testing procedures are identical to VCGbench， where all evaluation scripts are [here](script/inference/qabench).

For instance, to evaluate the result on MSVD, we first run
```
bash script/inference/qabench/msvd_qa.sh
```
and then run
```
bash script/inference/qabench/score_msvd.sh
```



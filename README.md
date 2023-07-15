# Towards Practical Plug-and-Play Diffusion Models （CVPR 2023） 
[[Arxiv]](https://arxiv.org/abs/2212.05973)  [[Open Access]](https://openaccess.thecvf.com/content/CVPR2023/html/Go_Towards_Practical_Plug-and-Play_Diffusion_Models_CVPR_2023_paper.html) [[BibTex]](#BibTex)

Official Pytorch Implementation of the paper "Towards Practical Plug-and-Play Diffusion Models".
This repository contains the code for guidance with 1) Finetuned models on forward diffused data 2) Multi-Expert strategy 3) PPAP, which are used in the paper.


This repository is based on following repositories with some modifications: 
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [microsoft/LoRA](https://github.com/microsoft/LoRA)


## Plan
- [x] Release code.
- [ ] Make checkpoints available.


## Requirements
For distributed training, [MPICH](https://www.mpich.org/) should be installed with following commands.
```
apt install mpich
pip install git+https://github.com/openai/CLIP.git --no-deps
```

For installing required python packages, use this commands.
```
pip install -r requirements.txt 
```


## Imagenet Class Guidance for ADM

### A. Prepare pre-trained diffusion models.
For the pre-trained diffusion model, we use ADM which trained on imagenet 256x256 dataset.
Checkpoint of this model is available at  [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt).

Download it and save on the path `[diffusion_path]`.


### B. Train
Our code supports training 1) finetuned model 2) multi-experts 3) PPAP. 
Here is commands for these.

1. Finetune off-the-shelf models on forward diffused data. 
    ```
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    MODEL_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 1e-4 --weight_decay 0.05 --save_interval 10000"
    CLASSIFIER_FLAGS="--image_size 256 --classifier_name [classifier name: ResNet18, ResNet50, ResNet152, DEIT]"
    python python_scripts/classifier_train.py --log_path [directory for logging] --data_dir [ImageNet1k training dataset path] --method "finetune" $MODEL_FLAGS $CLASSIFIER_FLAGS --gpus 0
    ```
2. Multi-Experts that are supervisely trained.
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   MODEL_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 1e-4 --weight_decay 0.05 --save_interval 10000"
   CLASSIFIER_FLAGS="--image_size 256 --classifier_name [classifier name: ResNet18, ResNet50, ResNet152, DEIT]"
   python python_scripts/classifier_train.py --log_path [directory for logging] --data_dir [ImageNet1k training dataset path] $MODEL_FLAGS $CLASSIFIER_FLAGS --gpus 0 --n_experts [Number of experts] --method "multi_experts"
   ```
3. PPAP.
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   MODEL_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 1e-4 --weight_decay 0.05 --save_interval 10000"
   CLASSIFIER_FLAGS="--image_size 256 --classifier_name [classifier name: ResNet18, ResNet50, ResNet152, DEIT] --lora_alpha 8 --gamma 16"
   python python_scripts/classifier_train.py --log_path [directory for logging] --data_dir [ImageNet1k training dataset path] $MODEL_FLAGS $CLASSIFIER_FLAGS  --gpus 0 --n_experts [Number of experts] --method "ppap"
   ```

#### B.1 Enabling DDP for training
If mpich is installed, distributed data parallel (DDP) can be enabled for training.
For DDP with `k` gpus, `--batch_size` should be divided by `k`, `mpiexec -n k` should be specified in front of python execution command, and `--gpu` option should be set by gpu ids that will be used.

For example, above finetuning off-the-shelf models with DDP on `0, 1, 2, 3` gpus can be executed with following commands:
```
 export PYTHONPATH=$PYTHONPATH:$(pwd)
 MODEL_FLAGS="--iterations 300000 --anneal_lr True --batch_size 64 --lr 1e-4 --weight_decay 0.05 --save_interval 10000"
 CLASSIFIER_FLAGS="--image_size 256 --classifier_name [classifier name: ResNet18, ResNet50, ResNet152, DEIT]"
 mpiexec -n 4 python python_scripts/classifier_train.py --log_path [directory for logging] --data_dir [ImageNet1k training dataset path] --method "finetune" $MODEL_FLAGS $CLASSIFIER_FLAGS --gpus 0 1 2 3
```


### C. Sampling with classifier guidance
Our code supports sampling with guidance from 1) finetuned model 2) multi-experts 3) PPAP.

1. Finetune
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   SAMPLE_FLAGS="--batch_size 100 --num_samples 10000  --timestep_respacing ddim25 --use_ddim True"
   MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
   MODEL_PATH_FLAGS="--model_path [diffusion_path] --classifier_path [ckpt_path]"
   python python_scripts/classifier_sample.py --log_path [sampling_path] $MODEL_FLAGS $SAMPLE_FLAGS $MODEL_PATH_FLAGS --method "finetune" --gpus 0
   ```
2. Multi-experts
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   SAMPLE_FLAGS="--batch_size 100 --num_samples 10000  --timestep_respacing ddim25 --use_ddim True"
   MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
   MODEL_PATH_FLAGS="--model_path [diffusion_path] --classifier_path [ckpt_path_0] [ckpt_path_1] ... [ckpt_path_N]"
   python python_scripts/classifier_sample.py --log_path [sampling_path] $MODEL_FLAGS $SAMPLE_FLAGS $MODEL_PATH_FLAGS --method "multi_experts" --gpus 0
   ```
3. PPAP
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   SAMPLE_FLAGS="--batch_size 100 --num_samples 10000  --timestep_respacing ddim25 --use_ddim True"
   MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
   MODEL_PATH_FLAGS="--model_path [diffusion_path] --classifier_path [ckpt_path_0] [ckpt_path_1] ... [ckpt_path_N]"
   python python_scripts/classifier_sample.py --log_path [sampling_path] $MODEL_FLAGS $SAMPLE_FLAGS $MODEL_PATH_FLAGS --method "ppap" --gpus 0
   ```

#### C.1 Sampling configuration.
1. DDIM: To sample by DDIM with `t` steps, set `--timestep_respacing ` as `ddimt`.
2. DDPM: DDPM with `t` steps is enabled when `--timestep_respacing` is set as `t`.


#### C.2 DDP for sampling.
Because of slow sampling speed, we recommend to use DDP for sampling. 
For using DDP with `k` gpus, please add command `mpiexec -n k` in front of python execution command, 
and set `--gpu` option to gpu ids that will be used.


### D. Evaluation
Check [evaluations/Readme.md](evaluations/Readme.md).



## PPAP with various models for DeepFloyd-IF.
### A. Prepare pre-trained model weight of DeepFloyd-IF.

### B. Generate unconditional image dataset for PPAP.

### C. Plug-in

## BibTex
```
@inproceedings{go2023towards,
  title={Towards Practical Plug-and-Play Diffusion Models},
  author={Go, Hyojun and Lee, Yunsung and Kim, Jin-Young and Lee, Seunghyun and Jeong, Myeongho and Lee, Hyun Seung and Choi, Seungtaek},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1962--1971},
  year={2023}
}
```
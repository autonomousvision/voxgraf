# VoxGRAF

<div style="text-align: center">
<img src="gfx/ffhq.gif" width="512"/><br>
</div>

This repository contains official code for the paper
[VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids](https://www.cvlibs.net/publications/Schwarz2022NEURIPS.pdf).

You can find detailed usage instructions for training your own models and using pre-trained models below.

If you find our code or paper useful, please consider citing

    @inproceedings{Schwarz2022NEURIPS,
      title = {VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids},
      author = {Schwarz, Katja and Sauer, Axel and Niemeyer, Michael and Liao, Yiyi and Geiger, Andreas},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2022}
    }

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create and activate an anaconda environment called `voxgraf` using

```commandline
conda env create -f environment.yml
conda activate voxgraf
```

### CUDA extension installation

Install pre-compiled CUDA extensions by running
```commandline
./scripts/build_wheels.sh
```
**Or** install them individually by running
```commandline
pip install dist/stylegan3_cuda-0.0.0-cp39-cp39-linux_x86_64.whl
pip install dist/svox2-voxgraf-0.0.1.dev0+sphtexcub.lincolor.fast-cp39-cp39-linux_x86_64.whl
pip install dist/MinkowskiEngine-0.5.4-cp39-cp39-linux_x86_64.whl       # optional, only required when training with minkowski sparse convolutions
```
In case the wheels do not work for you, you can also install the extensions from source. For this please check the original repos: [Stylegan-3](https://github.com/NVlabs/stylegan3), [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) and for our version of Plenoxels follow the instructions [here](voxgraf-plenoxels/README.md).

## Pretrained models
To download the pretrained models run
```commandline
./scripts/download_pretrained_models.sh
```

### Evaluate pretrained models
```commandline
# generate a video with 1x2 samples and interpolations between 2 keyframes each
python gen_video.py --network pretrained_models/ffhq256.pkl --seeds 0-3 --grid 1x2 --num-keyframes 2 --output ffhq_256_samples/video.mp4 --trunc=0.5

# generate grids of 3x4 samples and their depths
python gen_images.py --network pretrained_models/ffhq256.pkl --seeds 0-23 --grid 3x4 --outdir ffhq_256_samples --save_depth true --trunc=0.5
```

## Train custom models

### Download the data
Download [FFHQ](https://github.com/NVlabs/stylegan2), [AFHQ](https://github.com/clovaai/stargan-v2) and [Carla](https://github.com/autonomousvision/graf).

### Preparing the data
To prepare the data at the required resolutions you can run
```commandline
./scripts/make_dataset.sh /PATH/TO/IMAGES data/{DATASET_NAME}.json data/{DATASET_NAME} 32,64,128,256
```
This will create the datasets in `data/{DATASET_NAME}_{RES}.zip`.

### Train models progressively

```commandline
# Train a model on FFHQ progressively starting at image resolution 32x32 with voxel grid resolution 32x32x32
python train.py --outdir training-runs --gpus 8 --data data/ffhq_32.zip --batch 64 --grid-res 32
python train.py --outdir training-runs --gpus 8  --data data/ffhq_64.zip --batch 64 --grid-res 32 --resume /PATH/TO/32-IMG-32-GRID-MODEL                                    # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_64.zip --batch 64 --grid-res 64 --resume /PATH/TO/64-IMG-32-GRID-MODEL                                    # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_128.zip --batch 64 --grid-res 64 --lambda_vardepth 1e-3 --resume /PATH/TO/64-IMG-64-GRID-MODEL            # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_128.zip --batch 32 --grid-res 128 --lambda_vardepth 1e-3 --resume /PATH/TO/128-IMG-64-GRID-MODEL          # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_256.zip --batch 32 --grid-res 128 --lambda_vardepth 1e-3 --resume /PATH/TO/128-IMG-128-GRID-MODEL         # Next stage

# Train a model on Carla at image resolution 32x32 with voxel grid resolution 32x32x32
python train.py --outdir training-runs --gpus 8  --data data/ffhq_32.zip --batch 64 --grid-res 32 --n-refinement 0 --use_bg False --lambda_sparsity 1e-8
python train.py --outdir training-runs --gpus 8  --data data/ffhq_64.zip --batch 64 --grid-res 32 --n-refinement 0 --use_bg False --lambda_sparsity 1e-8 --resume /PATH/TO/32-IMG-32-GRID-MODEL                                    # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_64.zip --batch 64 --grid-res 64 --n-refinement 0 --use_bg False --lambda_sparsity 1e-8 --resume /PATH/TO/64-IMG-32-GRID-MODEL                                    # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_128.zip --batch 64 --grid-res 64 --n-refinement 0 --use_bg False --lambda_sparsity 1e-8 --lambda_vardepth 1e-3 --resume /PATH/TO/64-IMG-64-GRID-MODEL            # Next stage
python train.py --outdir training-runs --gpus 8  --data data/ffhq_128.zip --batch 32 --grid-res 128 --n-refinement 0 --use_bg False --lambda_sparsity 1e-8 --lambda_vardepth 1e-3 --resume /PATH/TO/128-IMG-64-GRID-MODEL          # Next stage
```
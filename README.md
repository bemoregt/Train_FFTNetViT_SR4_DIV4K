# Train FFTNetViT Super Resolution (4x) with DIV2K Dataset

This repository contains the implementation of a diffusion-based super-resolution model for 4x upscaling trained on the DIV2K dataset. The model is designed to work with MPS acceleration on Apple Silicon devices.

## Overview

This project implements a diffusion-based model for image super-resolution. The model uses a modified UNet architecture with enhanced residual blocks for better performance on high-resolution images. It's specifically trained for 4x upscaling.

## Features

- **MPS/CUDA/CPU Support**: Automatically selects available hardware acceleration (prioritizes MPS for Apple Silicon)
- **DIV2K Dataset Handler**: Includes complete dataset downloading and preparation functionality
- **Enhanced UNet Architecture**: Modified for high-resolution image processing
- **Diffusion-based Training**: Implements noise prediction with customizable diffusion steps
- **Visualization**: Sample generation and visualization during training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- tqdm
- PIL
- requests

## Model Architecture

The super-resolution model is based on a diffusion process and consists of:

1. **Enhanced UNet**: A feature extraction backbone with:
   - Time embedding for diffusion process
   - 3-stage downsampling path
   - Multiple residual blocks in the bottleneck
   - 3-stage upsampling path
   - Skip connections for better information flow

2. **Diffusion Model**: Implements the forward and reverse diffusion processes:
   - Forward diffusion: Gradually adds noise to images
   - Reverse diffusion: Progressively denoises images
   - Customizable noise schedule

## Training Process

The training process includes:

1. Loading and preprocessing the DIV2K dataset
2. Training the diffusion model on high-resolution patches
3. Periodic model checkpointing
4. Generating and visualizing samples during training

## Usage

To train the model:

```bash
python train_sr_diffusion.py
```

The script will:
1. Download the DIV2K dataset if not already present
2. Initialize the model
3. Train for the specified number of epochs
4. Save the model and generate sample outputs

## Super-Resolution Results

During training, the model generates sample super-resolution results in the `sr_samples` directory. These samples show the model's progress in learning to generate high-resolution images.

## Model Parameters

- **Image Size**: 256×256 pixels (high-resolution)
- **Scale Factor**: 4x upscaling
- **Hidden Channels**: 64
- **Batch Size**: 2 (optimized for memory constraints)
- **Learning Rate**: 1e-4
- **Diffusion Steps**: 1000

## License

MIT

## Acknowledgements

- DIV2K dataset from the NTIRE 2017 Challenge
- Diffusion model approach based on recent research in generative modeling

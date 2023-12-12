# U-Net Image Segmentation

## Overview
This repository contains a TensorFlow and Keras implementation of the U-Net architecture for image segmentation. U-Net is a popular convolutional neural network used in computer vision for tasks such as semantic segmentation.

## Description
The U-Net architecture consists of an encoder, bottleneck, and decoder. It is particularly effective for image segmentation tasks where detailed spatial information is crucial. The model is designed to take an input image and output a segmentation mask, highlighting regions of interest.

## Code Structure
- **conv2d_block**: Adds two convolutional layers with specified parameters and applies ReLU activation.
- **encoder_block**: Implements a convolutional block and performs downsampling on the output of convolutions.
- **encoder**: Defines the encoder or downsampling path by chaining multiple encoder blocks.
- **bottleneck**: Implements bottleneck convolutions to extract additional features before upsampling layers.
- **decoder_block**: Defines one decoder block of the U-Net, combining upsampling and skip connections.
- **decoder**: Connects four decoder blocks to reconstruct the segmented output.
- **unet**: Defines the entire U-Net architecture by connecting the encoder, bottleneck, and decoder.

## Usage
```python
# Instantiate the U-Net model
model = unet()

# View the model architecture
model.summary()
```

## Model Configuration
- Input Shape: (128, 128, 3)
- Output Channels: 3 (adjustable based on the number of classes)

## Dependencies
- TensorFlow
- Keras

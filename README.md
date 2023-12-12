# U-Net Architecture for Image Segmentation

## Overview
This repository contains the implementation of a U-Net architecture for image segmentation using TensorFlow and Keras. U-Net is a convolutional neural network commonly used for semantic segmentation tasks in computer vision.

## Code Structure

### `conv2d_block` function
```python
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # Adds 2 convolutional layers with specified parameters
    # ...
    return x
```

### `encoder_block` function
```python
def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    # Adds convolutional block and performs down sampling on the output of convolutions
    # ...
    return f, p
```

### `encoder` function
```python
def encoder(inputs):
    # Defines the encoder or downsampling path
    # ...
    return p4, (f1, f2, f3, f4)
```

### `bottleneck` function
```python
def bottleneck(inputs):
    # Defines the bottleneck convolutions to extract more features before the upsampling layers
    # ...
    return bottle_neck
```

### `decoder_block` function
```python
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    # Defines one decoder block of the U-Net
    # ...
    return c
```

### `decoder` function
```python
def decoder(inputs, convs, output_channels):
    # Defines the decoder of the U-Net chaining together 4 decoder blocks
    # ...
    return outputs
```

### `unet` function
```python
def unet():
    # Defines the U-Net architecture by connecting the encoder, bottleneck, and decoder
    # ...
    return model
```

## Usage
```python
# Instantiate the model
model = unet()

# See the resulting model architecture
model.summary()
```

## Model Configuration
- Input shape: (128, 128, 3)
- Output channels: 3 (adjustable based on the number of classes)

## Dependencies
- TensorFlow
- Keras


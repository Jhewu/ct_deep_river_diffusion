"""
THIS SCRIPT CONTAINS THE NECESSARY COMPONENTS 
TO BUILD THE U-NET FOR THE DIFFUSION MODEL
"""

""" ALL IMPORTS """
# import necessary libraries
import tensorflow as tf
import keras
from keras import layers

# import from local scripts
from sinusoidal_embedding import SinusoidalEmbedding
from parameters import embedding_dims

"""-------------------------------------------------CODE BELOW------------------------------------------------------"""

"""
HIGH LEVEL SUMMARY: 
This function defines a residual block for a neural network
Residual blocks are layers which outputs are added to a deeper
layer
    - It checks the input width (input_width) of the tensor x.
    - If the input width matches the specified width, it sets the residual to be the same as x.
    - Otherwise, it applies a 1x1 convolution (layers.Conv2D) to transform the input tensor to the desired width.
    - Next, it applies batch normalization (layers.BatchNormalization) and two 3x3 convolutions with ReLU activation
        (swish activation is used here).
    - Finally, it adds the residual tensor to the output tensor and returns it.
"""
@keras.saving.register_keras_serializable()
def ResidualBlock(width): # width specify the number of output channels
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x # set residual to be the same as x if it matches
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x) # set residual to the desired width
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply

"""
HIGH LEVEL SUMMARY: 
This function defines a downsampling block that reduces the spatial dimensions of the input tensor.

    - It expects a tuple (x, skips) as input, where x is the input tensor, and skips is a list to 
        store intermediate tensors.
    - It repeatedly applies block_depth residual blocks to the input tensor.
    - After that, it performs average pooling (reducing spatial dimensions) on the output tensor.
    - The function returns the downsampled tensor.
"""
@keras.saving.register_keras_serializable()
def DownBlock(width, block_depth): 
    # width is number of output channels for the residual blocks
    # block_depth determines how many residual blocks are stacked in this 
        # downsampling block
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x) 
            # average pooling reduces spatial dimensions
        return x
    return apply
        # returns the downsampled tensor

"""
HIGH LEVEL SUMMARY:
This function defines an upsampling block that increases the spatial dimensions of the input tensor.
        - It also expects a tuple (x, skips) as input.
        - It first performs upsampling using bilinear interpolation (layers.UpSampling2D).
        - Then, it concatenates the upsampled tensor with the last tensor stored in skips.
        - It applies block_depth residual blocks to the concatenated tensor.
        - The function returns the upsampled tensor.
"""
@keras.saving.register_keras_serializable()
def UpBlock(width, block_depth):
    # same parameters as downblock with width and block_depth
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x) # upsampling here
        for _ in range(block_depth):
            a = skips.pop()
            # print("this is the concatenate (layer, skip):",  x, "and", a)
            x = layers.Concatenate()([x, a]) # concatenates the upsampled tensor with
            x = ResidualBlock(width)(x)                # the last tensor stored in skips (a stack)
        return x
    return apply
        # returns upsampled tensor

"""
HIGH LEVEL SUMMARY: 
    - Creates U-Net Model
        - The model takes inputs [noisy_images, noise_variances] and produces the denoised output.
        - The model is named “residual_unet”.
    - Uses the few functions mentioned above
    - Check comments for more information
"""
@keras.saving.register_keras_serializable()
def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size[0], image_size[1], 3)) # Input for noisy images
    noise_variances = keras.Input(shape=(1, 1, 1))                # Input for noise variances

    #print("this is noisy images", noisy_images)

    sinusoidal_layer = SinusoidalEmbedding(embedding_dims) 
        # from our custom Sinusoidal Embeddig Layer 
    
    # Call the layer with your input (e.g., noise_variances)
    e = sinusoidal_layer(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
        # noise variances inputwidthed to sinusoidal layer
        # then to upsampling using nearest neighbor interpolation

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])
        # noisy images input into Conv2D 
        # output is concatenated with e

    skips = [] # skip is a list
    for width in widths[:-1]:
        #print("this is downblock width", width)
        x = DownBlock(width, block_depth)([x, skips])
            # series of downblocks are applied to the concatenated features
            # each downblock reduces spatial resolution and increases the number
                # of filters
        #("this is downblock shape", x)

    for _ in range(block_depth):
        #print("this is the bottleneck width" , widths[-1])
        x = ResidualBlock(widths[-1])(x)
            # stack of residual blocks is applied
        #print("this is bottleneck shape", x)

    for width in reversed(widths[:-1]):
        #print("this is upblock width", width)
        x = UpBlock(width, block_depth)([x, skips])
            # each block upsamples the features and reduces the number 
                # of filters
        #print("this is upblock shape", x)

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
        # final convolution, 1x1 convolution with 3 channels (RGB) is applied to the output

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


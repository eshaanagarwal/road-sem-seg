import network_architectures.pix2pix as pix2pix
import tensorflow as tf


def unet_model(IMG_SIZE, output_channels):
    """
    This method creates a modified U-net with a MobileNetV2 encoder.
    :param IMG_SIZE: The size of input images to expect
    :param output_channels: how many classes are there to distinguish between
    :return: a model architecture
    
    """

    # Load a pretrained MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMG_SIZE, 3], include_top=False)

    # Use the activations of only these layers, instead of the whole model in order to preserve symmetry between
    # U-net's contracting and expansive paths.
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False  # Freeze the layers in the contracting path of the model

    # Get the expansive path using upsampling layers represented by Conv2DTranspose layers
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    # Get input layer
    inputs = tf.keras.layers.Input(shape=[*IMG_SIZE, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model, where we decide how many elements the pixel-wise one-hot vectors should have
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)  # return generated model

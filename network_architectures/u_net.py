from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model


def unet(input_size, _optimizer, _loss, _metrics, num_class=2):
    """
    A method that creates a U-Net model, compiles and returns it.
    The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture.

    The contracting path is a typical convolutional network that consists of a repeated application of
    convolutions, rectified linear unit (ReLU) and a max pooling operations. During the contraction, the
    spatial information is reduced while feature information is increased - i.e., the images shrink in width and
    
    height but their 3rd channel actually increases with depth.

    The expansive pathway combines the feature and spatial information through a sequence of upsampling
    -convolutions and concatenations with high-resolution features from the contracting path (skip connections).

    :param input_size: The shape of the input images - usually it is (Width, Heighth, Channels), where
    channels stands for color encoding. For example, Red-Green-Blue (RGB) has 3 channels, one for each color
    :param _metrics: A list of training metrics' names that will be monitored during the training - e.g. ['accuracy']
    :param _loss: Loss function that will be used during training
    :param _optimizer: A Keras compatible  optimizer object or the string encoded name of that object. For example,
                            -> k.opt.Adam() or 'adam' <-
    :param num_class: Number of distinct semantic object classes that will be presented in the labels
    :return: A Keras implementation of the U-net model architecture by Ronnenberg et al. (2015)

    *Reference to paper:
        Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical
        image segmentation. In International Conference on Medical image computing and computer-assisted intervention
        (pp. 234-241). Springer, Cham.
    """

    # Build the pipeline using keras.Layer.__call__() interface
    # Layers are stacked one on top of the other, hence the output layer is the top-most one

    inputs = Input(input_size)  # Input layer
    # Downsampling block 1
    # ---------------------------------- Contracting path symmetrical to expansive -------------------------------------
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Downsampling block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Downsampling block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Downsampling block 4 with Dropout of 0.5 - to reduce risk of overfitting
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # ------------------------------------------------ End -------------------------------------------------------------

    # ---------------------------- Middle element (deepest decoder layer) ----------------------------------------------
    # Downsampling block 5 again with a Dropout of 0.5
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # ------------------------------------------------ End -------------------------------------------------------------

    # ------------------------------ Expansive path reversely symmetrical to contracting -------------------------------
    # Upsampling 'resize-convolution' block 1
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)  # Merge outputs from symmetric contracting layer and this resize-conv
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Upsampling 'resize-convolution' block 2
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Upsampling 'resize-convolution' block 3
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Upsampling 'resize-convolution' block 4
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # ------------------------------------------------ End -------------------------------------------------------------

    # Generate the output layer and fit everything into:
    #                           - one channel in case of Binary Classes - x < 0.5 for class 1 and x >= 0.5 for class 2
    #                           - number of classes channels, otherwise

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # 1 x 1 convolution to do the pixel-wise label assigning
    model = Model(inputs=inputs, outputs=conv10)

    # Compile model and print summary
    model.compile(optimizer=_optimizer, loss=_loss, metrics=_metrics)  # compile model
    print('Printing summary for model...')
    model.summary()  # print summary

    return model  # return model architecture


def unet_transposed(input_size, _optimizer, _loss, _metrics, num_class=2):
    """
    A method that creates a U-Net model, compiles and returns it.
    The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture.

    The contracting path is a typical convolutional network that consists of a repeated application of
    convolutions, rectified linear unit (ReLU) and a max pooling operations. During the contraction, the
    spatial information is reduced while feature information is increased - i.e., the images shrink in width and
    height but their 3rd channel actually increases with depth.

    The expansive pathway combines the feature and spatial information through a sequence of TRANSPOSED CONVOLUTIONAL
    layers and concatenations with high-resolution features from the contracting path (skip connections).

    :param input_size: The shape of the input images - usually it is (Width, Heighth, Channels), where
    channels stands for color encoding. For example, Red-Green-Blue (RGB) has 3 channels, one for each color
    :param _metrics: A list of training metrics' names that will be monitored during the training - e.g. ['accuracy']
    :param _loss: Loss function that will be used during training
    :param _optimizer: A Keras compatible  optimizer object or the string encoded name of that object. For example,
                            -> k.opt.Adam() or 'adam' <-
    :param num_class: Number of distinct semantic object classes that will be presented in the labels
    :return: A Keras implementation of the U-net model architecture by Ronnenberg et al. (2015)
    """
    inputs = Input(input_size)
    # Downsampling Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # Downsampling Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # Downsampling Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # Downsampling Block 4
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Middle Block
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    """ 
    We must replace the conv-upsample block with conv2dtranspose layer
    The conv2d layer is applied to the output of the upsample layer which is applied
    to the drop5 output
    8 x 8 x 1024 (output of drop5) to 16 x 16 x 512 (output of up6)
    """
    # Up-Convolutional Block 4 - with Merge
    # 'up6' takes input 8 x 8 x 1024 transforms it into 16 x 16 x 1024 as the upsampler does and
    # then applies convolution and reduces its filters from 1024 to 512 (Final shape: [16,16,512])
    up6 = Conv2DTranspose(512, 3, 2, padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Up-Convolutional Block 3 - with Merge
    up7 = Conv2DTranspose(256, 3, 2, padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Up-Convolutional Block 2 - with Merge
    up8 = Conv2DTranspose(128, 3, 2, padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Up-Convolutional Block 1 - with Merge
    up9 = Conv2DTranspose(64, 3, 2, padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = _loss

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=_optimizer, loss=loss_function, metrics=_metrics)
    model.summary()

    return model

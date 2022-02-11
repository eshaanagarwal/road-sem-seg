import os

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


## Helpers

def show_img(img, title=None):
    """
    A method that simply displays an image on screen.
    :param img: Image data - can be a numpy array, a tensor, PIL.Image
    :param title: A title for the plot
    """
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()


def generate_true_and_predicted_masked_img_and_overlay(test_image, pred_mask, real_mask, mdl_version):
    """
    A method that generates RGB masks from predicted segmentations and ground truths.
    It also overlays the images on top of the original testing image to show overlaps.
    :param test_image: The real-size raw testing image
    :param pred_mask: A grayscale 1-channel PIL.Image of the predicted segmentation mask !rescaled to [0..255]!
    :param real_mask: A grayscale 1-channel PIL.Image of the annotated target mask !rescaled to [0..255]!
    :param mdl_version: Specifier for the model type that is being used to produce the segmentation mask
    :return: A tuple with 3 RGB images -
                    'real_lane_img' is the RGB ground truth mask,
                    'pred_lane_img' is the RGB predicted mask,
                     'result' is the RGB test image that contains both masks overlaid on top of it showing overlapq
    """
    # transform the masks into numpy array and convert all pixel values to integers
    mask = np.array(pred_mask).astype('uint8')
    real_mask = np.array(real_mask).astype('uint8')

    # Depending on the model apply conditioning operations.
    #   For example, in case of model 'unet':
    #       all pixel values in the mask arrays that are less than 150 are replaced by True,
    #       while all the larger values by False.
    #   Then, all booleans are converted into their equivalent integer representations (0 and 1)
    # In the encoded masks, the pixels containing a 1 are given the value 255 (prior to RGB conversion)
    if mdl_version == 'unet':
        mask = (mask < 150).astype('uint8')  # assign each pixel True or False depending on the condition (pixel < 150)
        real_mask = (real_mask < 150).astype('uint8')
    else:
        mask = (mask >= 1).astype('uint8')
        real_mask = (real_mask < 150).astype('uint8')
    mask[mask == 1] = 255  # put 255 where there is 1 in the mask array
    real_mask[real_mask == 1] = 255
    # Reference for this code https://stackoverflow.com/questions/19766757/replacing-numpy-elements-if-condition-is-met

    new_mask = mask.reshape(*mask.shape, 1)  # Reshape the arrays from (W,H) to (W,H,1) shapes
    new_real_mask = real_mask.reshape(*real_mask.shape, 1)  # Reshape the arrays from (W,H) to (W,H,1)

    # Generate fake R & G color channels, which will be stacked with B(lue) from the converted masks
    # Create a duplicate array of the mask with all values initialized to 0, it will have shape (W,H,1)
    blanks = np.zeros_like(new_mask).astype(np.uint8)

    # Use numpy to stack the image arrays one on top of the blanks. This creates an array with shape (W,H,3),
    # where the 3 channels are 0(blanks) for Red and Green and mask_values for Blue. This allows for the creation of
    # an image representation where only the pixels predicted as Road class have values different than 0, hence
    # are displayed when the original image and the mask are overlaid together
    pred_lane_img = np.dstack((blanks, blanks, new_mask))
    real_lane_img = np.dstack((blanks, blanks, new_real_mask))

    # Here, we stack the real and predicted masks together on the same raw image.
    # The real mask is in red, while the other in blue -
    #       you can tell by their order in the tuple below (Red, Green, Blue)
    # This creates an image where the Red channel has the expected class labels and the Blue one has the predicted ones
    lane_img = np.dstack((new_real_mask, blanks, new_mask))

    # Then, merge the lane drawing onto the original image
    result = cv2.addWeighted(np.array(test_image), 1, lane_img, 1, 0)

    return [real_lane_img, pred_lane_img, result]


def prepare_for_prediction(img, size):
    """
    Used to resize images, convert into numpy arrays and add a 4th dimention to their shapes.
    This 4th dimension is for the batch_size. As the model is trained on batches of data, it expects
    an input with shape (N,W,H,1), where N is batch size, W is width, H is height and 1 is the channel dimension
    :param img: A test image that will be predicted by the model - it is a three channel RGB PIL.Image
    :param size: The size (width,height) that the neural network expects as input
    :return: returns a normalized numpy array scaled to [0..1] and with an exoanded first dimension
    """
    img_arr = np.array(img.resize(size)) / 255  # normalize and convert to ndarray
    img_arr = np.expand_dims(img_arr, axis=0)  # add a batch dimension
    return img_arr


def interpret_prediction(pred_img, size):
    """
    This method converts a prediction into an grayscale PIL.Image.
    It receives U-NETs predictions which are 4 dimensional arrays with values scaled to [0..1] and removes their first
    (batch) dimension. Then all pixel values are multiplied by 255 and rescaled between [0..255]
    :param pred_img: A 4-dimensional predicted numpy array
    :param size: Size to which the prediction has to be resized to after formatting
    :return: A 1-channel PIL.Image converted from the predicted numpy array
    """
    converted_img = np.reshape(pred_img, pred_img.shape[1:3])  # drop batch dimension
    converted_img = converted_img * 255
    return (Image.fromarray(converted_img)).resize(size)


def create_mask(pred_mask):
    """
    This method is for MobileNetV2's predictions. It applies argmax function that preserves the highest value
    in the predicted pixel-wise one-hot vectors. That is, if pixel p has value [0.3, 0.7] the output of argmax() will
    be [0.7].
    :param pred_mask: a predicted numpy array
    :return: A 1-channel PIL.Image converted from the predicted numpy array (the method scales it to [0..255])
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return tf.keras.preprocessing.image.array_to_img(pred_mask[0], scale=True)


class DisplayCallback(tf.keras.callbacks.Callback):
    """
    This class is a callback method that after each epoch predicts a test image and displays the prediction on screen.
    That way the progress of the model can be seen during the training itself.
    """

    def __init__(self, model_type, _model, test_img_path, data_home, label_dir, img_size):
        """
        Initialiser method that creates the objects needed for the callback to function
        :param model_type: the type of model - 'unet' or 'mobnet'
        :param _model: the currently trained model after the most recent epoch. It will predict the chosen test image
        :param test_img_path: the path to the test image that will be used in the prediction
                              can also be a LIST of paths for multiple images
        :param data_home: the parent directory of the whole data set. It will be used to load labels
        :param label_dir: the directory containing labels within data_home.
        :param img_size: what size the images need to be in order to fit into the model architecture
        """
        super().__init__()  # Keras callback parent class stuff
        self.type = model_type
        self._model = _model
        self._img_size = img_size

        if type(test_img_path) != list:  # If only a single image path is given, then put it into a list
            test_img_path = [test_img_path]

        self.test_imgs = {}  # Dict that will hold image names as keys and a pair of PIL.Images (img, label)
        for img_path in test_img_path:  # for all images in the given test img paths list
            im = Image.open(img_path)  # open
            test_lbl_name = img_path.rpartition('/')[-1].rpartition('_')  # get img filename from path
            # derive label name from img
            test_lbl_name = test_lbl_name[0] + test_lbl_name[1] + 'road_' + test_lbl_name[-1]
            test_lbl = Image.open(
                f'{os.path.join(data_home, label_dir)}/' + test_lbl_name
            ).convert('L')  # open label
            self.test_imgs[img_path.rpartition('/')[-1]] = (im, test_lbl)  # add key:value pair to dict
        self.test_img_names = list(self.test_imgs.keys())  # get a list of all image names available for predicting

    def on_epoch_end(self, epoch, logs=None):
        """
        When calling Kerases' fit() method, we can specify callbacks. They all have to implement this on_epoch_end()
        method so that at the end of every epoch Kerases' API can call it.
        :param epoch: The current epoch number
        :param logs: Some log data for debugging
        :return: None
        """
        for name in self.test_img_names:
            show_img(self.test_imgs[name][0], title=name)  # Show the raw image before predicting
            test_img, test_lbl = self.test_imgs[name]  # get the label and img PIL.Image data from the dict
            test_arr = prepare_for_prediction(test_img, self._img_size)  # convert image to a numpy array
            prediction = self._model.predict(test_arr)  # predict image using the current version of the model

            if self.type == 'unet':  # If the model is U-NET
                # Convert numpy array to 1-channel PIL.Image
                resulting_img = interpret_prediction(prediction, test_img.size)

            elif self.type == 'mob_net':  # If the model is MobileNetV2
                # Call the equivalent conversion method
                resulting_img = create_mask(prediction).resize(test_img.size)

            else:
                resulting_img = test_arr

            # Display the predicted segmentation mask
            show_img(resulting_img, title=f'Predicted image for test img: {name}')
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))  # log info for me


def get_saved_models_and_choice(save_dir="models/", ensemble=False):
    """
    A method that takes user input and allows us to choose a pretrained model we want to load into memory.
    Note: The models' directories need to have the word 'model' in their names
    :param ensemble: This is a boolean specifier that indicates if we are in ensemble mode and several models will
                     need to be loaded.
    :param save_dir: The directory to look into.
    :return: A string encoded path to model.
    """
    # Get all model directories that start with 'model' and are directories :D
    _models = [
        name for name in os.listdir(save_dir)
        if name[:5] == 'model' and os.path.isdir(os.path.join(save_dir, name))
    ]
    if not _models:
        return -1
    for j, mdl in enumerate(_models):  # Print a list of all models with appropriate indices
        print(f"\t{j + 1} - {mdl}")

    if ensemble:
        while True:
            print("Choose models from here that will be loaded for ensemble.")
            choices = input("Type their indices separated by spaces here:\n\t-> ").split()
            try:  # Get all model names chosen by user
                ensemble_models = [
                    os.path.join(save_dir, _models[int(model_index)-1]) for model_index in choices
                ]
                return ensemble_models
            except IndexError:
                print('One of the model indices provided is not valid.\n'
                      'Please use only the numbers you see on in the list provided above.\n')
                continue
            except ValueError:
                print('One of the provided values is not a number.\n'
                      'Please use only the numbers you see on in the list provided above.\n')
                continue
    else:
        print("Choose a model from here and type its index in the input space bellow:")
        while True:  # Get user input and handle exceptions
            try:
                choice = int(input("\tType choice here -> "))
                break
            except ValueError:
                print('Please provide a number. Not any other character.')
                continue

        return os.path.join(save_dir, _models[choice - 1])  # return path to model


def pretty(d, indent=0, title=''):
    """
    A method for printing dictionary hierarchy.
    :param d: a dictionary object to iterate
    :param indent: how many tabs to use to indent each level of data
    :param title: a title for the printed dictionary
    :return: None
    """
    if title:
        print(title)
        print('_' * 100)
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


## Tests

def test_for_repetitions(data_gen):
    """
    This method tests whether the partial sampling algorithm was successful - i.e., the images added to training for
    transfer learning from the test data are not gonna be presented to model during inference time.
    :param data_gen: The data generator object that will be used.
    :return: The result from test - a boolean.
    """
    print('\nTesting for repetitions (not wanted images) in test and train separation from the same directory...')
    # Get a testing sample generator
    testing_gen = data_gen.get_testing_data()
    testing_pairs = list(testing_gen)  # Extract all pairs of testing images from generator and put them to a list

    res = False  # Initially we assume it works
    # Get only the image names from the image-label pairs as we will look for repetitions there
    ab, db = [name.rpartition('/')[-1] for name, _ in testing_pairs], \
             [name.rpartition('/')[-1] for name, _ in data_gen.data]
    for ae in ab:
        if ae in db:
            res = True  # Keep checking if each image from testing set is in training set
            break
    print("Result of the test:", 'Not passed' if res else 'Passed', '\n')
    return res

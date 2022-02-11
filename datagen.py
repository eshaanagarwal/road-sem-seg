import silence_tensorflow.auto
from tensorflow import keras
import numpy as np
import os
from PIL import Image
import math

from tqdm import tqdm


class DataGen(keras.utils.Sequence):
    """
    A subscriptable class to generate batches of training data in the form of numpy arrays. The generated and returned data is
    a tuple with two tuples - one for the image batch and one for the label batch. This class also creates two
    generator objects that yield validation samples and testing samples.


    Data shape for training batches is:
        ((image_batch), (label_batch)).
    Each batch contains N images/labels with (W,H,C) dimensions - width, height, depth(channels), respectively.
    """

    def __init__(self, data_home, train_dir, label_dir, test_dir, batch_size, img_size, model_type, val_split=0.1,
                 augmentation_data='', partial_sampling=0.0):
        """
        Initializer method of the generator class
        :param data_home: the home directory of the data. It has to contain the train_dir, label_dir and test_dir. If
        augmented data is used, then it has to reside in this parent dir, too.
        :param train_dir: the subdirectory residing in data_home which contains TRAINING SAMPLES
        :param label_dir: the subdirectory residing in data_home which contains TRAINING LABELS
        :param test_dir: the subdirectory residing in data_home which contains TESTING IMAGES
        :param augmentation_data: the subdirectory residing in data_home that contains AUGMENTED TRAINING SAMPLES
        :param batch_size: the number of image-label pairs to load on one __get_item__() call.
        i.e. DataGen_Object[0] generates a tuple with two batch_size tuples (the image batch and label batch)
        :param img_size: The size to which images/labels will be resized prior to training or validation
        :param model_type: Verison of the model that will be used on the data 'unet' or 'mob_net'. This determines how
        the each pixel in the original labels is being encoded - i.e., as a one-hot vector showing probabilities for
        the pixel to belong to each class (MobileNetV2) or simply with one class id, indicating which class the pixel
        belongs to (for U-NET)
        :param val_split: Percentage of the total training data set to split for validation:
            (total_samples * val_split) = number of validation samples separated from the main training data
        :param partial_sampling: This indicates what fraction of the training data to actually use for training. It is
        useful for small data sets used with Transfer Learning since it allows the model to be trained only on some % of
        the training data. The rest of the samples are preserved for evalutaion.
        """

        # PATHS
        self.data_path = data_home
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.test_dir = test_dir
        self.aug_dir = augmentation_data

        # PARAMS
        self.BATCH_SIZE = batch_size
        self.model_type = model_type
        self.img_size = img_size
        self.partial_sampling = bool(
            partial_sampling)  # This variable is used in the testing generator to check whether
        # partial sampling has been requested and executed. It ensures that the returned testing samples have not been
        # used during the training.

        # Data references
        self.data = self.get_image_names_and_labels()  # returns a list of tuples [(img_path, label_path), (...), ...]
        np.random.shuffle(self.data)  # We randomly shuffle the data to ensure thar validation and/or partial sampling
        # will always result in different sets
        if self.partial_sampling:  # If partial sampling has been requested slice the main data set so that it includes
            # only the first m sample-label pairs. The rest will be used during evaluation.
            self.data = self.data[:math.ceil(len(self.data) * partial_sampling)]
        if val_split > 0.0:
            self.val_data = self.data[-math.ceil(len(self.data) * val_split):]  # take % for validation data
            self.validation_generator = self.load_batch_size_val_data()  # get a generator object for validation samples
            self.data = self.data[:-math.ceil(len(self.data) * val_split)]  # the rest of the data is left for training

    def __len__(self):
        """
        Method to return the length of the data generator.

        :return: It returns the number of batches that will be generated from the given number of samples
        """
        return math.ceil(len(self.data) / self.BATCH_SIZE)

    def __getitem__(self, index):
        """
        This method returns a batch of images and a batch of corresponding labels everytime DataGen_Object[index]
        is executed. The returned images and labels are normalized and resized. Labels are also encoded depending
        on the neural model's requirements.

        :param index: The batch index that is being requested from the model.fit() method call
        :return: A tuple of numpy arrays - (image_batch, label_batch) containing batch_size number of elements
        """
        image_batch, label_batch = list(), list()

        # for each sample-target pair in the 'indexth' batch of data
        for pair in self.data[self.BATCH_SIZE * index:self.BATCH_SIZE * (index + 1)]:
            # Open images from the specified path as PIL.Image objects
            image = Image.open(os.path.join(self.data_path, pair[0]))
            label = Image.open(os.path.join(self.data_path, pair[1]))
            # Normalize, resize and encode the images and labels as required by the model type
            normalized_img, normalized_label = self.normalize_pair((image, label), self.img_size, self.model_type)
            # Append each instance into the corresponding batch
            image_batch.append(normalized_img)
            label_batch.append(normalized_label)
        # Convert list() objects to numpy arrays and return for training
        return np.array(image_batch), np.array(label_batch)

    @staticmethod
    def normalize_pair(img_label_pair, img_size, version):
        """
        A static class method that does what its name says

        :param img_label_pair: A pair of training image and its raw label
        :param img_size: The size that the products will be resized to
        :param version: A model that is currently being used for training or will be used
        :return: A tuple of two normalied, resized, encoded numpy arrays - img and label
        """
        img_arr = np.array(img_label_pair[0].resize(img_size)) / 255  # scale pixel values from [0..255] to [0..1]
        label_raw = np.array(img_label_pair[1].resize(img_size)).astype('uint8')  # 8-bit ints to enable quicker exec.
        label_arr = list()

        # Different encoding is applied depending on the model version used
        if version == 'unet':
            # Encode each white pixel from the label image with the value of 1 (class Non-road)
            # and every other with the value of 0 (class Road)
            for i, row in enumerate(label_raw):
                new_row = [[1] if np.array_equal(pixel, np.array([255, 255, 255])) else [0] for pixel in row]
                label_arr.append(np.array(new_row))
        elif version == 'mob_net':
            # Encode each white pixel with the one-hot vector [1,0] and every other pixel with [0,1]
            # The first index indicates the Non-road class and the second the Road class
            # So white pixels ([1,0]) are treated as [Non-road:True, Road:False] and vice versa
            for i, row in enumerate(label_raw):
                new_row = [[1, 0] if np.array_equal(pixel, np.array([255, 255, 255])) else [0, 1] for pixel in row]
                label_arr.append(np.array(new_row))
        label_arr = np.array(label_arr)  # convert list to numpy array
        return img_arr, label_arr

    def load_batch_size_val_data(self):
        """
        A generator method that loads "batch_size" number of images and labels from the validation set into memory.
        Since each validation sample must be in the same format as the training samples,
        the normalize_pair() method is called.
        The while loop will ensure that the generator keeps yielding samples forever.
        """
        val_img, val_label = list(), list()  # Lists that will hold the current img and label batches
        print(
            f"\nPreparing validation generator for {len(self.val_data)} validation samples and starting validation...")
        counter = 0  # This counter will guide the while loop by tracking the number of loaded samples
        batch_counter = 0  # Simply counts which batch is being loaded now
        while True:
            print(f'Loading batch {batch_counter+1}..', end='\r')
            # Load images, normalise and append to current validation batch
            image = Image.open(os.path.join(self.data_path, self.val_data[counter][0]))
            label = Image.open(os.path.join(self.data_path, self.val_data[counter][1]))
            normalized_img, normalized_label = self.normalize_pair((image, label), self.img_size, self.model_type)
            val_img.append(normalized_img)
            val_label.append(normalized_label)

            # The counter tracks
            #       1. if a batch_size number of samples are generated - WE HAVE TO YIELD THE BATCH and empty lists
            #       2. whether all validation samples have been exhausted and there are no more data - WE HAVE TO YIELD
            #          THE LOADED SAMPLES, reset all lists and set the counter to 0 as we will go over the data again.
            counter += 1
            # on final batch (may be incomplete - i.e., can have less samples than the batch_size)
            if counter == len(self.val_data):
                yield np.array(val_img), np.array(val_label)  # yield all remaining samples
                counter = 0  # reset counter back to start of val_list
                val_img = []
                val_label = []
                print('Finished loading the last batches.')
            elif counter % self.BATCH_SIZE == 0:  # if a batch_size validation batch is loaded and normalised, yield it
                batch_counter += 1
                yield np.array(val_img), np.array(val_label)
                val_img = []  # reset batch lists
                val_label = []

    def get_testing_data(self, seen_images=None):
        """
        A generator object that yields a string encoded path to a testing image and its label
        The self.test_dir is used in this method to get all test file names while self.label_dir for labels.
        This method also ensures that samples used by partial sampling are not reused for evaluation
        """
        if seen_images is None:
            seen_images = []
        # Sort images by name for convenience
        test_img_names = sorted(
            [img_name for img_name in os.listdir(os.path.join(self.data_path, self.test_dir))
             if img_name not in seen_images]
        )
        if self.partial_sampling and self.train_dir == self.test_dir:  # if partial sampling is used and
            # test and train data sets are the same, exclude the training images from testing list - i.e., keep only
            # the images that the model has not seen during transfer learning.
            to_exclude = [name.rpartition('/')[-1] for name, _ in self.data]
            test_img_names = [test_im for test_im in test_img_names if test_im not in to_exclude]

        print('\nGenerating testing data...')
        for img_name in tqdm(test_img_names):  # After the list of paths to testing images is prepared
            # Get the ground_truth label for the current testing image and return it along with its corresponding image
            # Labels are returned to allow the segmentation accuracy to be measured by comparing the label
            # with the prediction.
            tmp = img_name.rpartition('_')
            label_name = ''.join(tmp[:2]) + 'road_' + tmp[2]
            yield os.path.join(self.data_path, self.test_dir, img_name), \
                  os.path.join(self.data_path, self.label_dir, label_name)

    def get_image_names_and_labels(self):
        """
        This method is used to iterate through the training and labels directories and generate a list with
        sample-target pairs as tuples. It assumes that the label for image "umm_000001.png" is "umm_road_000001.png".
        The same is valid for augmented data. The only difference is that those image and label names begin with "new_".
        :return: a list of 2-tuples containing paths to image-label pairs.
        """
        image_names = os.listdir(os.path.join(self.data_path, self.train_dir))  # list all image names in the train dir
        name_pairs = []
        print('\nGetting image and label filenames and adding them to DataGen\'s data attribute...')
        for name in tqdm(image_names):  # iterate through all raw images
            tmp = name.split('_')  # 'um_000011.png'.split('_') = ['um','000011.png']
            # Append '_road_' in the middle to derive the corresponding ground-truth label
            label_name = tmp[0] + '_road_' + tmp[1]
            # Add the train_dir name to the image name and the label name and append the resulting paths into the names
            #   list as a tuple (img_path, label_path) - e.g. 'um_road' + '/' + 'um_road_*.png' = um_road/um_road_*.png
            name_pairs.append((self.train_dir + '/' + name, self.label_dir + '/' + label_name))

            if self.aug_dir:  # If we requested the use of augmented samples
                # The same procedure as above, but here we also add 'new_' to the image an label name - 'new_um_*.png'
                name_pairs.append((self.aug_dir + '/' + 'new_' + name, self.label_dir + '/' + 'new_' + label_name))

        return name_pairs  # we return the list of image-label pairs

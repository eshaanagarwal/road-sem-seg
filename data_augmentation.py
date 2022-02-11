import argparse

import Augmentor
import numpy as np
import os
from PIL import Image
import math
from tqdm import tqdm


def create_new_augmented_samples(train_direc, label_dir, output_dir):
    """
    This method is used to alter the original training images and their labels and create artificial data samples.
    The Augmentor library is used to apply the modifications. It has a wide range of transformation and also ensures
    that labels of images are modified exactly identically to preserve consistency in training data.
    :param train_direc: a path to the directory where all training samples reside
    :param label_dir: a path to the directory where all the corresponding labels reside
    :param output_dir: a path to the directory where the generated augmented IMAGES will be saved;
                       the LABELS are saved in the label dir by default.
                       
    """
    # create output dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create a list of all image filenames in the training directory
    images_to_augment = [name for name in os.listdir(train_direc) if '.png' in name]
    labels_to_augment = []
    for img in images_to_augment:
        img_name = img.rpartition('_')
        label_name = img_name[0] + '_road_' + img_name[-1]
        labels_to_augment.append(label_name)

    # If the current number of augmented images
    data = list(zip(images_to_augment, labels_to_augment))
    print('\nFirst 3 image-label pairs to augment:')
    [print(f'\t{pair}') for pair in data[:3]]

    print('\nAugmenting samples... Depending on the number of examples this can take some time.')
    print(f'The average expected execution time is {math.ceil((len(images_to_augment) / 2.5) / 60)} minutes.')
    for image_name, label_name in tqdm(data):
        # Load the image and its label
        image = np.array(Image.open(os.path.join(train_direc, image_name)))
        label = np.array(Image.open(os.path.join(label_dir, label_name)))

        # Create an Augmentor Pipeline instance and initialise it with the sample pair.
        # The pipeline will apply modification based on probability one by one
        p = Augmentor.DataPipeline([[image, label]])
        p.flip_left_right(1)
        p.rotate(1, 2.5, 2.6)
        p.zoom(1, 1.35, 1.55)
        samples = p.sample(1)  # create one sample - a list of N lists => [[sample1, sample1'], [sample2, sample2']]
        new = 'new_'
        Image.fromarray(samples[0][0]).save(os.path.join(output_dir, new + image_name))
        Image.fromarray(samples[0][1]).save(os.path.join(label_dir, new + label_name))

    print(f'\nNew files have been created and saved to path {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train_dir',
        help='This should be the path to the train images you want to augment. \n'
             'Please make sure this path contains the data_home directory, too.\n',
        type=str
    )
    parser.add_argument(
        'label_dir',
        help='This should be the path to the label masks you want to augment. \n'
             'Please make sure this path contains the data_home directory, too.\n',
        type=str
    )
    parser.add_argument(
        'output_dir',
        help='This should be the path to the directory where the augmented images \n'
             'will be saved. If the directory doesn\'t exist, it will be created along \n'
             'with all corresponding parent directories in the path. \n',
        type=str
    )

    args = parser.parse_args()
    create_new_augmented_samples(
        train_direc=args.train_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir
    )


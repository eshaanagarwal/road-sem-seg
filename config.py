"""
This is a configuration file which contains some global objects used elsewhere

"""
import argparse
from helpers import pretty
import pickle

if __name__ == '__main__':
    # Create an argument parser to allow modifications to configuration parameters from command line/terminal
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--imsize',
                        help='This defines the size of the images inputted to the network.\n'
                             'The data must be a tuple of two elements (Width, Height).\n'
                             'NOTE: Model will not work if the given size is not compatible with its Input Layer.',
                        type=tuple, default=(128, 128))
    parser.add_argument('-b', '--bsize',
                        help='The number of samples to load in each batch '
                             '(the available sizes are shown in curly the brackets - {2,4,...,32}).\n'
                             'Depending on the capacity of the GPU or RAM, large "bsize" might terminate execution.',
                        type=int, choices=[2, 4, 8, 16, 32], default=4)
    parser.add_argument('-c', '--classes',
                        help='Number of object classes the model needs to choose from for each pixel.',
                        type=int, default=2)
    parser.add_argument('-e', '--epochs',
                        help='Number of training epochs.',
                        type=int, default=20)
    parser.add_argument('-mv', '--model_version',
                        help='The model architecture that will be used for training.',
                        type=str, choices=['unet', 'mob_net'], default='unet')
    parser.add_argument('-v', '--val_split',
                        help='What fraction of the data to use for validation during training.',
                        type=float, metavar="[0.0, 1.0]", default=0.2)
    parser.add_argument('-s', '--steps_per_epoch',
                        help='Number of batches to iterate over in each epoch.',
                        type=int, default=None)
    parser.add_argument('-a', '--augmented_data',
                        help='Path to the augmented images. The corresponding augmented labels\n'
                             'are expected to be in the main ground-truth folder.',
                        type=str, default='training/new_augs')
    parser.add_argument('-p', '--partial_sampling',
                        help='What fraction of the data to use for partial sampling during transfer learning.',
                        type=float, metavar="[0.0, 1.0]", default=0.0)
    parser.add_argument('-d', '--data_home',
                        help='Path to directory containing all training and testing sample subdirectories.',
                        type=str, default='KITTI/data_road')
    parser.add_argument('-tr', '--train_dir',
                        help='Path to the training images within the "--data_home" directory',
                        type=str, default='training/um_road')
    parser.add_argument('-l', '--label_dir',
                        help='Path to ground-truth labels within "--data_home".\n'
                             'This directory must include also the augmented images` labels.',
                        type=str, default='enc_gt_image_1')
    parser.add_argument('-tt', '--test_dir',
                        help='Path to the testing samples within "--data_home".',
                        type=str, default='training/umm_road')
    parser.add_argument('-ms', '--model_save_dir',
                        help='Name of the directory where the trained model an its weights will be saved.',
                        type=str, default='models/')
    parser.add_argument('-cf', '--config_filename',
                        help='The name of the file that will be used to save the serialised (pickled) dictionary'
                             ' object. Do not add a file extension - a .pickle one will be added anyway.',
                        type=str, default='current_config')

    args = parser.parse_args()

    # Save all arguments to dict
    params = {
        'IMG_SIZE': args.imsize,
        'BATCH_SIZE': args.bsize,
        'OUTPUT_CHANNELS': args.classes,
        'EPOCHS': args.epochs,
        'VERSION': args.model_version,
        'VAL_SPLIT': args.val_split,
        'STEPS_PER_EPOCH': args.steps_per_epoch,
        'AUGMENTATION_DATA': args.augmented_data,
        'PARTIAL_SAMPLING': args.partial_sampling,
        'data_home': args.data_home,
        'train_dir': args.train_dir,
        'label_dir': args.label_dir,
        'test_dir': args.test_dir,
        'model_save_dir': args.model_save_dir
    }

    # Print dictionary and then save it to a pickle file for later loading
    pretty(params, title='Current values for all configuration parameters.')
    print(f'\nCreating a pickle file [{args.config_filename}.pickle] from config dictionary...')
    with open(f'{args.config_filename}.pickle', 'wb') as pickler:
        pickle.dump(params, pickler)
    print('File has been created successfully.')




import silence_tensorflow.auto
from tensorflow import keras
import argparse
import math
from focal_loss import BinaryFocalLoss
from datagen import DataGen
from helpers import *
from network_architectures.u_net import unet
from network_architectures.mobile_net_v2 import unet_model
import pickle


def train_model(_model, _data_gen, _epochs, _callbacks, _save_name,
                _steps_per_epoch=None, _save_dir='saved_models/'):
    """
    This method is a wrapper for the main Keras API model.fit() method that:
        . checks variables and config params,
        . creates a model save directory and saved the model and its best set of weights
        . generates loss and accuracy plots and saves them as PDF to the model save directory
        . saves the History.history object that is returned by .fit() as a numpy file
        . saves all image names used during training in a text file for later references
    :param _model: The neural network architecture to be trained - either U-net or MobileNetV2.
    :param _data_gen: A data generator object that will return batches of data encoded appropriately for the certain
                      model that's being trained. It also needs to have a validation data generator and generate batches
                      of validation samples (image-label pairs).
    :param _epochs: The number of epochs to train the model for (unless 'early stopping' is engaged).
    :param _callbacks: The list of callback objects to call at the end of every epoch - all must implement the
                       on_epoch_end() class method.
    :param _save_name: The name to save the model with. More data will be appended in the end like current date, time,
                       loss and accuracy scores, etc., to ensure model names are unique and no overwriting occurs.
    :param _steps_per_epoch: This is allows us to specify how many batches of data we want to iterate over. Leave 'None'
                             if all the data should be used to train the model.
    :param _save_dir: The name for the directory where the trained model will be saved.
                      If this directory does not exist, it will be created.
    :return: A Keras History.history object that hold information about the training process and metric results
             and the name of the directory the model was saved to
             
    """
    if _save_dir[-1] == '/':  # Just to ensure path is valid
        _save_dir = _save_dir[:-1]

    if not os.path.exists(_save_dir):  # If the save directory does not exist,
        # create it
        os.makedirs(_save_dir)

    # Call Kerases' .fit() method
    # The validation_steps are used to determine how many batches the validation generator will yield before exhausting
    # all available samples in the validation set. The ceiling rounding operation ensures that the final partially full
    # batches will also be considered by the program.
    model_history = _model.fit(x=_data_gen,
                               epochs=_epochs,
                               validation_data=_data_gen.validation_generator,
                               validation_steps=math.ceil(len(_data_gen.val_data) / _data_gen.BATCH_SIZE),
                               callbacks=_callbacks,
                               steps_per_epoch=_steps_per_epoch
                               )

    # Load date and time, which will be used to create a nonce-like info
    from datetime import datetime, date

    now = datetime.now()
    today = date.today()

    # Get date and time in appropriate formats
    current_date = today.strftime("%b-%d-%Y")
    current_time = now.strftime("%H-%M-%S")

    # Create a string-encoded path where the model will be saved. Also create its unique name here, too.
    model_save_path = f"{_save_dir}/{_save_name}_{current_date}_{current_time}_imgsize_{params['IMG_SIZE']}_" \
                      f"epochs_{_epochs}_val_acc_{model_history.history['val_accuracy'][-1]:.4f}_" \
                      f"val_loss_{model_history.history['val_loss'][-1]:.4f}"

    # Generate plots for loss/val_loss and accuracy/val_accuracy using the history object
    plt.figure(12, figsize=(6, 6), dpi=60)
    plt.subplot(211)
    plt.plot(model_history.history['loss'], label='train')
    plt.plot(model_history.history['val_loss'], label='val')
    plt.title('loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(model_history.history['accuracy'], label='train')
    plt.plot(model_history.history['val_accuracy'], label='val')
    plt.title('accuracy')
    plt.legend()

    # Save the model to the above created path
    _model.save(model_save_path)
    # Save a PDF file containing both plots - for loss and for accuracy
    plt.savefig(os.path.join(model_save_path, 'loss_and_accuracy.pdf'), transparent=True, dpi=1200)
    # Save the History.history object (it is a dict) as a numpy file
    np.save(os.path.join(model_save_path, 'training_history.npy'), model_history.history)
    # Save a txt file that contains all image names used for training
    filenames = [name.rpartition('/')[-1] for name, _ in _data_gen.data if 'new_' not in name]
    with open(f'{model_save_path}/used_images.txt', 'w') as f:
        for name in filenames:
            f.write(f'{name}\n')

    print("Model saved successfully!")
    return model_history, model_save_path  # Return History


if __name__ == '__main__':
    # This argument parser is responsible for handling commandline arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--config_filepath',
        help='Path to a pickled dictionary object (include the .pickle extension, too) containing all comfiguration '
             'params. \nThe config.py file generates such a file if it is not present in current directory.',
        type=str, default='default_config.pickle'
    )
    parser.add_argument(
        '-v', '--visualise',
        help='Decide whether to use DisplayCallback() object',
        action='store_true'
    )
    # Get values of arguments - if none are given, the defaults are gonna be used
    args = parser.parse_args()

    # Try to open the pickled config file. If an error arises abort execution and notify user
    try:
        with open(args.config_filepath, 'rb') as pickled_dict:
            params = pickle.load(pickled_dict)
    except FileNotFoundError as err:
        print(f'The following error occured during execution:\n\t-> {err}\n'
              f'Please define a valid path to a pickled config dict. It can be generated by exicuting:\n\t'
              f'>>> python config.py | for help run python config.py -h')
        exit()

    # Create a data generator object from variables in the config.py file
    data_gen = DataGen(
        data_home=params['data_home'],
        train_dir=params['train_dir'],
        label_dir=params['label_dir'],
        test_dir=params['test_dir'],
        batch_size=params['BATCH_SIZE'],
        img_size=params['IMG_SIZE'],
        model_type=params['VERSION'],
        val_split=params['VAL_SPLIT'],
        augmentation_data=params['AUGMENTATION_DATA'],
        partial_sampling=params['PARTIAL_SAMPLING']
    )

    assert test_for_repetitions(data_gen) is False, f"There has been an error in the parital sampling algorithm.\n" \
                                                    f"Some images are both in training and testing sets."

    # Define and compile basic model - Normal U-NET Focal LOSS and train
    LOSS = BinaryFocalLoss(gamma=0.5)  # a Keras compatible loss function
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)  # a Keras compatible optimizer object

    model = unet(input_size=(*data_gen.img_size, 3),
                 _optimizer=OPTIMIZER,
                 _loss=LOSS,
                 _metrics=["accuracy"],
                 num_class=params['OUTPUT_CHANNELS'])

    # Define a model save dir name
    model_save_name = input(
        "Enter a name for the name your trained model will be saved with. "
        "(Example: 'model_AUG_um_umm_unet_kitti')\n\t --> "
    )

    # Create callback methods
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f'{params["model_save_dir"]}/{model_save_name}.hdf5',
        monitor='loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    callbacks = [model_checkpoint, early_stopping]
    # If we want visualisations, create a callback to create them and add it to the list of callbacks
    if args.visualise:
        display_callback = DisplayCallback(
            data_gen.model_type,
            _model=model,
            test_img_path=['KITTI/data_road/training/umm_road/' + image for image in
                           ['umm_000024.png', 'umm_000086.png', 'umm_000053.png']],
            data_home=params['data_home'],
            label_dir=params['label_dir'],
            img_size=params['IMG_SIZE']
        )
        callbacks.append(display_callback)

    # Train the model using wrapper method
    history, full_model_dir_name = train_model(
        _model=model,
        _data_gen=data_gen,
        _epochs=params['EPOCHS'],
        _callbacks=callbacks,
        _save_name=model_save_name,
        _steps_per_epoch=params['STEPS_PER_EPOCH'],
        _save_dir=params['model_save_dir'],
    )

    print(f'\nModel training finished. You can access the trained model from:\n\t'
          f'-> {full_model_dir_name}')

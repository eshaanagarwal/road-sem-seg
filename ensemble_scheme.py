import numpy as np
from helpers import show_img
from PIL import Image
import os


def apply_voting_to_ensemble_predictions(predictions, savepath=None):
    """
    This method merges the predictions of multiple models by applying majority hard voting. It takes the first
    prediction and scales it down to [0..1] from [0..255] (normalisation) to avoid overflow of pixels -
        
    After that, the remaining predictions are also normalised. A pixel-wise addition is peformed between all predicted
    masks. The result is an image where each pixel holds a value which is the sum of all values of this pixel
    in all predictions - (pixel_i = [0, 0, 1] + pixel_j = [0, 0, 1]) = pixel_k = [0, 0, 2].

    These pixel values (the sums) are then averaged (divided by the total number of ensembles) which gives the final
    values. The prediction mask is then conditionised - all values larger than 0.5 are set to 0 (BACKGROUND because
    black in the mask) while all the rest are set to 1 (ROAD class because will be scaled to 255).
    :param savepath: a path to save the intermediate masks to
    :param predictions: a list of predicted segmentation masks. They have to be numpy arrays with the same shape (W,H,1)
    :return: an averaged prediction computed by applying 'HARD PIXEL-WISE MAJORITY VOTING' scheme to all predictions.
    """
    # initialize the ensemble prediction image and normalize it

    voted_pred = predictions[0] / 255
    if savepath:
        Image.fromarray(predictions[0]).convert('RGB').save(os.path.join(savepath, 'prediction_1.png'))
    else:
        show_img(predictions[0])
    for j, pred in enumerate(predictions[1:]):
        if savepath:
            Image.fromarray(pred).save(os.path.join(savepath, f'prediction_{j+2}.png'))
        else:
            show_img(pred)
        voted_pred = voted_pred + (pred / 255)

    averaged_pred = voted_pred / len(predictions)
    if savepath:
        Image.fromarray((voted_pred*255).astype(np.uint8)).save(os.path.join(savepath, 'voted_prediction.png'))
        Image.fromarray((averaged_pred*255).astype(np.uint8)).save(os.path.join(savepath, 'averaged_prediction.png'))
    else:
        show_img(voted_pred)
        show_img(averaged_pred)
    averaged_pred = (averaged_pred < 0.501).astype(np.uint8)

    return averaged_pred

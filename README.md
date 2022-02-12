# Semantic segmentation :

Semantic segmentation is the process of classifying each pixel belonging to a particular label. It doesn't different across different instances of the same object. For example if there are 2 cats in an image, semantic segmentation gives same label to all the pixels of both cats.

**Different Usecases** : Autonomous vehicles and Medical Diagnosis

# Road Segmentation for ADAS :
In recent years, growing research interest is witnessed in automated driving systems (ADS) and advanced driver assistance systems (ADAS). 
In the ADS workflow, an accurate and efficient road segmentation is necessary. 

A drivable region is a connected road surface area that is not occupied by any vehicles, pedestrians, cyclists or other obstacles. 

Convolutional neural network (CNN) based algorithms attracted research interest in recent years. Existing CNN based road segmentation algorithms such as FCN, SegNet, StixelNet, Up-conv-Poly and MAP generated a precise drivable region but required large computational. 

Recent research proposed several efficient networks and network structures such as MobileNet and Xception. So, here, is the road segmentation work of Ensemble based Unet and MobileUnetV2 model.


# Quick Instructions of Using the Code :
(For more details check out instructions to run code pdf)

## Set-Up Working Environment - 

### Setting up the virtual env :

 Run the following codes in your terminal to set up the virtual env.

```bash	
$ conda create -n myenv python=3.9
$ source ~/miniconda3/etc/profile.d/conda.sh
$ conda activate myenv
$ conda deactivate   # deactivate env.
```

Install dependencies from requirements.txt file
```bash
$ pip install -r requirements.txt
```
---------------------------------------------------------
### Model Training :

Run the following command if default pickle used 
```bash
$ python training.py -c default_config.pickle
```

Or 

```bash
$ python training.py -c --config_file_path	
```
--------------------------------------------------------

### Evaluation on Training Dataset :
```bash
$ python evaluation.py	
```
Or 	
```bash
$ python evaluation.py -c --config_file_path
```
------------------------------------------------------------
### Testing our Model on random image : 
```bash
$ python inference.py
```
Or 	
```bash
$ python inference.py -c --config_file_path
```

# INSightR-Net

This repository contains the code for INSightR-Net (accepted for MICCAI 2022). For any problems with the code please open an issue or email: linde.hesse@seh.ox.ac.uk. 

## Requirements

All requirements are given in requirements.txt 

## Dataset

The dataset used in this study can be downloaded from https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data. 

For pre-processing as done in this paper, the script preprocess_images/preprocess_BGraham.py can be used. This script is based on the pre-processing used by the Kaggle competition winner (https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801). To run this script, add the path of the folder containing 'train' and 'test' folders in line 66.  

The datasplits used for this study are given in config/datasplit, and demonstrates the structure that is expected when running this for a new dataset.

## Implementation

The INSightR-Net can be trained using

    python main_trainer.py --param_jsonpath=config/params_example_ordinal.json --datapath=datapath --savepath=savepath --pretrained_path=pretrained_model.ckpt

with datapath the path containing the pre-processed images, savepath the path where to save the trained model, and pretrained_path the path to a pretrained model checkpoint. The script can also be run without a pretrained model, but this did not result in convergence in our experiments. The pretrained model used in this study can be downloaded under 'releases' in the repository.

Param_json path should contain the parameter file used for training. All parameters used in this study are defined in 'define_parameters.py', all of these parameters can be overwritten in a .json file. The parameters varied in this study are given in 'params_example_ordinal.json' (for ordinal labels) and 'params_example_continuous.json' (for continuous labels, given in supplementary). 

A pre-trained model on other data can be obtained using the baseline model trainer. This script can be called using: 

    python main_trainer_baseline.py --param_jsonpath=config/params_example_baseline.json --datapath=datapath --savepath=savepath

For the experiment presented in the paper, we trained our baseline starting from the same pretrained_path as the INSightR-Net. This can be replicated using:

    python main_trainer_baseline.py --param_jsonpath=config/params_example_baseline.json --datapath=datapath --savepath=savepath --pretrained_path=config/pretrained_model.ckpt


## Logging

The script generate logging files in mlflow. These can be seen by running mlflow ui from the main code folder. 







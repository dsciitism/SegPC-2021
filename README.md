# SegPC-2021

This is the official repository for the ISBI 2021 paper __Transformer Assisted Convolutional Neural Network for Cell Instance Segmentation__ by Deepanshu Pandey, Pradyumna Gupta, Sumit Bhattacharya, Aman Sinha, Rohit Agarwal.

## Getting Started
We recommend using Python 3.7 for running the scripts in this repository. The necessary packages can be installed using _requirements.txt_ in the respective folders.

To clone this repository:

 ``` $ git clone https://github.com/dsciitism/SegPC-2021```

 
## Cascade Mask RCNN 

For installation of required packages:

``` $ pip install -r Cascade_Mask_RCNN_X152/requirements.txt ```


## DetectoRS

Preparation script should be run with the following command before running any other file in the DetectoRS folder :

``` $ bash mmdetection_preparation.sh ```

For installation of required packages:

``` $ pip install -r DetectorRS/requirements.txt ```

### Train

The following flags need to be used while running DetectoRS_train.py:

```bash

usage: DetectoRS_train.py [-h] --backbone {Original,Effb5,Transformer_Effb5} --train_data_root TRAIN_DATA_ROOT --training_json_path TRAINING_JSON_PATH [--train_img_prefix TRAIN_IMG_PREFIX] [--train_seg_prefix TRAIN_SEG_PREFIX] --val_data_root VAL_DATA_ROOT --validation_json_path VALIDATION_JSON_PATH [--val_img_prefix VAL_IMG_PREFIX] [--val_seg_prefix VAL_SEG_PREFIX] --work_dir WORK_DIR [--epochs EPOCHS] [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --backbone {Original,Effb5,Transformer_Effb5}
                        The backbone to be used from the given choices
  --train_data_root TRAIN_DATA_ROOT
                        path to training data root folder
  --training_json_path TRAINING_JSON_PATH
                        path to the training json file in COCO format
  --train_img_prefix TRAIN_IMG_PREFIX
                        prefix path ,if any, to be added to the train_data_root path to access the input images
  --train_seg_prefix TRAIN_SEG_PREFIX
                        prefix path ,if any, to be added to the train_data_root path to access the semantic masks
  --val_data_root VAL_DATA_ROOT
                        path to validation data root folder
  --validation_json_path VALIDATION_JSON_PATH
                        path to validation json file in COCO format
  --val_img_prefix VAL_IMG_PREFIX
                        prefix path ,if any, to be added to the val_data_root path to access the input images
  --val_seg_prefix VAL_SEG_PREFIX
                        prefix path ,if any, to be added to the val_data_root path to access the semantic masks
  --work_dir WORK_DIR   path to the folder where models and logs will be saved
  --epochs EPOCHS
  --batch_size BATCH_SIZE

```

*Note: DetectoRS requires semantic masks along with instance masks during training , hence the arguments - train_seg_prefix and val_seg_prefix*

### Inference

The following flags need to be used while running DetectoRS_inference.py:

```bash

usage: DetectoRS_inference.py [-h] --backbone {Original,Effb5,Transformer_Effb5} --saved_model_path SAVED_MODEL_PATH --input_images_folder INPUT_IMAGES_FOLDER --save_path SAVE_PATH

optional arguments:
  -h, --help            show this help message and exit
  --backbone {Original,Effb5,Transformer_Effb5}
                        The backbone to be used from the given choices
  --saved_model_path SAVED_MODEL_PATH
                        path to the saved model which will be loaded
  --input_images_folder INPUT_IMAGES_FOLDER
                        path to the folder where images to inference on are kept
  --save_path SAVE_PATH
                        path to the folder where the generated masks will be saved

```

## Citation
 
 

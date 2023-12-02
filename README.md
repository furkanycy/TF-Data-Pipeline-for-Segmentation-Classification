# TensorFlow Data_Pipeline For Segmentation and Classification of Images

This repository contains a TensorFlow data pipeline for artifact and tumor detection in histopathology images. The pipeline is designed to work with the Segmentation Models library, which provides a high-level API for training deep learning models for image segmentation and classification tasks. This repository is designed to speed up emprical process for model training for the task "artifact and tumor detection in histopathology images".

TODO:    

- [ ] Add code for "patching" data pipeline to work with large histopathology images.

- [ ] The classification part of the code will be added, the patching strategy is used here to work with large histopathology images.

## Data Structure
The data for this project should be structured as follows:

dataset\

 \images
 
 \labels
 
 \test_images
 
 \test_labels
 
images: This directory should contain the images for the training set. The validation will be created here as well.
labels: This directory should contain the corresponding labels for the images in the training set.
test_images: This directory should contain the images for the test set.
test_labels: This directory should contain the corresponding labels for the images in the test set.

## Usage
To use the data pipeline, you need to:

- Structure your data as described above.
- Install the required Python packages by running pip install -r requirements.txt.
- Run the example_segmentation.ipynb notebook to see how to use the data pipeline.

## Notes
Currently, this repository only contains a binary semantic segmentation pipeline. The classification part of the repository is yet to be developed. 

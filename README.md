# Deep Discriminating Hashing for Remote Sensing Image Retrieval

## Requirments
- cuda 9.0
- cudnn >= 7.0.1
- python >= 3.5
- tensorflow >= 1.9.0
- opencv-python >= 3.4.5.20

## Preparation
- Download HSRI Database from https://drive.google.com/drive/folders/1Qb4xjSB6PJRsoVy_m6253wsmFAykPCkD?usp=sharing
- Copy the *.txt files* from HSRI dataset to this folder (remember to change the path of each image).
- Put the pre-trained model of resnet-50 named *"resnet_v2_50.ckpt"* in the *models* folder (download from http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz).

## Training and Testing
Run *train.py* with the following parameters: *learning rate, alpha, theta, lambda, epoch, dimension of codes, batch size*.
(Example: *python train.py 0.001 0.0001 1.1 1 32 128*)

After training, the testing results will be given.

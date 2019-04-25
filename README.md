# Deep Discriminating Hashing for Remote Sensing Image Retrieval

## Prepare Data
- Download HSRI Database from https://drive.google.com/drive/folders/1Qb4xjSB6PJRsoVy_m6253wsmFAykPCkD?usp=sharing
- Copy the *.txt files* from HSRI dataset to this folder (remember to change the path of each image).
- Put the pre-trained model of resnet-50 named *"resnet_v2_50.ckpt"* in the *models* folder (download from http://download.tensorflow.org/models/resnet\_v2\_50\_2017\_04\_14.tar.gz).

## Training and Testing
Run *train.py* with the following parameters: *learning rate, alpha, theta, lambda, epoch, dimension of codes, batch size*.
(Example: *python train.py 0.001 0.0001 1.1 1 32 128*)
After training, the testing results will be given.

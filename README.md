# DDH
Deep Discriminating Hashing for Remote Sensing Image Retrieval

HSRI Database is available at https://drive.google.com/drive/folders/1Qb4xjSB6PJRsoVy_m6253wsmFAykPCkD?usp=sharing

Make three data files: train.txt, test.txt and database.txt containing the path of data for training and testing (see the format in the .txt file of HSRI Database). 
Put the pre-trained model of resnet-50 named "resnet_v2_50.ckpt" in the models folder (http://download.tensorflow.org/models/resnet\_v2\_50\_2017\_04\_14.tar.gz).

Training and testing model:
Run train.py with the following parameters: learning rate, alpha, theta, lambda, epoch, dimension of codes, batch size.
(Example: python train.py 0.001 0.0001 1.1 0.5 32 128)

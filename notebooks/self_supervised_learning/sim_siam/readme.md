# Simple Siamese Representation Learning

## Source

Exploring Simple Siamese Representation Learning by Xinlei Chen, Kaiming He, submitted on 20 Nov 2020
https://arxiv.org/abs/2011.10566

## Description

SimSiam is a non-contrastive technique, which means that only positive image pairs are used for training. In other words, data augmentation is applied to the same image and the network is supposed to detect the similarity between the images. No negative samples - a pair of two completely different images - are being used for the training process. In order to ensure that the network on the one hand learns a meaningful representation of the images and on the other hand does not always assume full similarity, a stop-gradient method is applied. The objective to be achieved is an optimal generalization of the network without learning too specific features of individual images, while not allowing the network to predict exclusively 100% similarity regardless of the data augmentation applied.

The encoder in SimSiam is based on a ResNet-50 architecture consisting of residual block groups consisting of a convolution layer, a batch normalisation layer and a ReLU activation. An average pooling layer is subsequently added, followed by two dense and two batch normalisation layers. The output of the first part of the network up to the average pooling layer is used as an input for the final classifier. This final classifier is based on a k-nearest neighbour classifier.

The results of the different training methods are displayed in the jupyter notebooks.

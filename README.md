# deep-dreaming-mini-me

## Introduction

DeepDream is developed by Google in order to transform normal images into kind of dreamy and phychedelic pictures. It works by enhancing the gradients with respect to activations of a set of layers and then adding these gradients to the image so that the features or patterns learned by the model will be enhanced.  

## How it works

To understand what DeepDream is, we first need to explain how a convolutional neural network works. CNNs use filters to extract patterns from an image. Filters are matrixes and convolution is basically an elementwise multiplication between two matrixes. Another operation in CNNs is pooling operation. Pooling is the reduction of resolution of images so that the image will contain fewer details. This operation is done between convolutional layers so that the next convolutional layers will extract larger patterns than the previous one. Thus both low level and high level patterns can be extracted from the image. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/210231306-c6edb1bd-79d0-4c21-a6ed-f4c8cce254a5.png"/>
</p>

DeepDream is result of an experiment where we enhance some of the patterns of the images and adding them to the original image to generate the dreamy pictures. If you want more details step by step approach can be seen below.

## Approach

1. Initialized the base model which is a pretrained model called Inception from tf.keras.applications without its classifier
2. Third and fifth layers of the model were used to extract the activations from the frame
3. Loss is calculated as the sum of the average of activations
4. The gradient is calculated by differentiating the loss by the frame
5. Gradient is normalized by dividing to its standard deviation and adding an epsilon value of 1e-8
6. Then the multiplication result of gradient and step size is added to the frame by a given number of steps

## Result

On left, the default video can be seen, on the center is the video deep dream processed by 100 steps and 0.01 step size and finally on the right the video deep dream processed by 250 steps and 0.01 step size.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/210204928-02cc3504-2bfb-4944-a7b1-ea925ba318d9.gif" width="250" height="250"/>
  <img src="https://user-images.githubusercontent.com/77073029/210204827-68fb463b-4ec9-4945-b82a-8bb4f866d6cd.gif" width="250" height="250"/>
  <img src="https://user-images.githubusercontent.com/77073029/210232563-dd182719-0d5e-4ff6-b8e4-0a8bb0343fa8.gif" width="250" height="250"/>
</p>


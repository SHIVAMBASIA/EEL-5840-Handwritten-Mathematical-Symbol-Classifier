# Final Project

This is a **group assignment**.

## Code Implementation & Technical Report

The final deliverables include a 4-page IEEE-format report, code implementation and a detailed GitHub readme file.

The final project is due Tuesday, December 6 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/455013/assignments/5244219) in the Canvas assignment.

## Training Data

The training data set is the same for every team in this course.

You can download the training data from the Canvas page:

* ["data_train.npy"](https://ufl.instructure.com/files/72247539/download?download_frd=1)
* ["t_train.npy"](https://ufl.instructure.com/files/72245951/download?download_frd=1)

## Edit this READ-ME

## Introduction

The Handwritten Mathematical Symbol Classifier is an image reader that identifies the mathematical symbols in a given image. This classifier identifies ten different symbols and classifies the rest of the symbols as unknown or undefined. The ten symbols identified by this classifier are:

* The letter x
* Square Root
* Plus sign
* Negative sign
* Equal sign
* Percentage
* Partial differential
* Product symbol
* pi 
* Summation

In addition to an image, the classifier also identifies the symbols in any given equation. This equation is given to the classifier as an image and can be handwritten or typed. The labels of the symbols are given in the order of their appearance in the equation. Any unknown symbol is labelled as "u". For ease of use, the images to be classified can be consolidated in a file and given as input. The final classifications of the images are also presented in a file. 


## Methodology

The Mathematical Symbol Classifier handles the classification in different ways depending on the input images. The approaches for each kind of input are as follows: 

### Individual Symbol Classification

For an accurate and a speedy classification of the symbols, this classifier uses CNN algorithm. The input images initially undergo pre-processing where they are first resized and then normalized. These pre-processed images, after being split into training and testing data, are then converted to tensors and divided into batches. The batches of images are given as input to the CNN model. 

The CNN model used here consists of three 2D convolutional layers, three pooling layers and three fully connected layers. ReLU activation function is used for the lower layers and softmax  activation is used for the output layer. 

Using the output of this model, the error is calculated. Depending on the error, the parameters used for the model are updated. The model is re-executed for the updated parameters. This process is repeated till the required number of epochs are executed. The overall training and validation accuracy are computed. The performance metrics used to evaluate this model are:

* Confusion matrix
* Accuracy
* Loss

The trained model is then stored for further usage.

### Symbol Classification In An Equation (Extra Credit)

In addition to the above classification, this classifier also identifies the symbols present in a given equation. These equations are given as images to the model. The model uses piecewise functions methodology to handle these images. 

The classifier initially detects contours in the image to identify the number of symbols in the equation. It then uses bounded rectangles to convert the image into multiple rectangles where each rectangle has a symbol. These rectangles are then trained in a linear fashion to figure out the labels of the symbols. Then these rectangles are merged back into one. The output would be the set of labels of all the symbols in the order that they are present in the equation. Any unknown symbol would be labeled as "u".  


### Hyperparameters used

* Pytorch is used for implementation
* Batch size is 10
* Number of epochs is 10
* Learning rate is 0.01 
* Threshold is 1e-7.
* Maximum batches are 3000 for both train and validation data
* Optimizer: Adam
* Loss function: Cross Entropy Loss
* Error: Mean Squared Error

## Steps to Execute

The following modules are required to run the project(The modules are already there in hipergator):

* torch
* sklearn
* cv2
* numpy
* pandas

The project has both .py and .ipynb files for execution. The steps to execute .ipynb files are as follows:

* Open Hipergator jupyter notebook using jhub.rc.ufl.edu
* Use the NGC-PyTorch-1.9 kernel.
* Change the filename in the variables 'images' and 'labels'(which is just after the import statements)
* Execute the notebook.

The steps to execute the .py files are as follows:

* For training: python train.py images-file-name labels-file-name
* For testing: python test.py images-file-name labels-file-name
* For testing the extra credit part: python test-extra-credit.py images-file-name labels-file-name

The training model is already saved in 'saved-model.pt'. So, the test file can be directly executed. The preprocessing and relabeling was done in Homework 3 Part 2 ,the result of which is t-train-corrected.npy which contains the data set with corrected labels.

## Results 

The model generates an output file 'output.csv' containing the original and predicted output for easy analysis of the model in the testing part. If the input is equations, the output file 'output-equation.csv' is generated and it contains the original file equation as well as the predicted equation.
Also,'output-equation-length.csv' contains the actual number of symbols in the equation as well as the predicted number of symbols in the equation.

**The training accuracy of the model is 95%. The validation accuracy of the model is 85%.**


## Conclusion and Future Scope

The offline mathematical symbol classifiers uses CNN algorithm to classify the ten mathematical symbols both individually and in equations. The model classifies the symbols with 95\% accuracy. Various algorithms were experimented on the data set to determine the best possible algorithm. Further, various hyper paramters and learning rates were used to find the best combination of hyper parameters. The best learning rate was identified to be 0.01. The output of the model are stored in csv files for easy access. 

This application can be further developed to recognize numbers and solve the equations provided in the images. 

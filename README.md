# Final-Project-Group1
Final Project ML2-Deep Learning
- Galaxy Zoo Image Classification - using CNN architecture and PYTORCH as Framework.

## Problem Statement:
To develop a Neural Network to classify Galaxy Images for 37 different categories with probabilities ranging between 0 to 1. As we have to classify 37 different labels for each image of the galaxy so its a Multi label Classification Problem.

## Approach
### To make the network manageable for training and validation or testing preprocessinng steps are performed:
#### Original Image - 424 x 424 
#### Center Crop - 256 x 256
#### Image Shape - 3 x 64 x 64  (RGB)
#### Target labels - One-hot encoded using Multi-label Binarizer

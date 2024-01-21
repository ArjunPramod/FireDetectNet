# FireDetectNet

FireDetectNet is an deep learning project that focuses on fire detection in UAV images using state-of-the-art convolutional neural networks (CNNs). This repository provides a comprehensive solution for training, evaluating, and deploying a robust fire detection model. My goal is to contribute to fire safety and emergency response efforts by automating the process of identifying fires in images. Whether you're interested in fire prevention or monitoring, FireDetectNet offers a powerful tool for image-based fire detection.

## Key Features

- **Deep Learning Model:** Utilizes a custom-designed CNN architecture for accurate fire detection.
- **Data Augmentation:** Enhances model robustness with data augmentation techniques.
- **Training and Evaluation:** Includes scripts for training the model and evaluating its performance.
- **Sample Images:** Provides sample images for quick model testing.
- **Visualization:** Offers tools to visualize training progress and model performance.
- **Easy Integration:** Seamlessly integrate the trained model into your applications for real-time fire detection.

## Getting Started

These instructions will help you get started with using and contributing to FireDetectNet. To begin, clone the repository to your local machine:

```bash
git clone https://github.com/ArjunPramod/FireDetectNet.git
cd FireDetectNet
```

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Visualization](#visualization)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction

This project focuses on detecting fires in UAV images using a convolutional neural network (CNN) model. The model is trained using a dataset of images containing both fire and non-fire scenes.

## Setup

1. Mount Google Drive:
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Set file paths:
   ```
   model_name = "/content/drive/MyDrive/1_FDM/Models/FDM_v1.h5"
   train_data_path = "/content/drive/MyDrive/1_FDM/Data/Train"
   test_data_path = "/content/drive/MyDrive/1_FDM/Data/Test"
   ```
3. Prerequisites
   Before you begin, ensure you have met the following requirements:
   - **Python 3**
   - **TensorFlow/Keras**
   - **OpenCV**
   - **NumPy**
   - **Matplotlib**
   - **Seaborn**
   
## Dataset
- The dataset is organized into train and test sets.
- Data loading and preprocessing functions are provided.

## Model Architecture
- The model architecture consists of convolutional layers, batch normalization, max-pooling, and dense layers.
- Dropout is applied to prevent overfitting.

## Training
- The train_and_save_fire_detection_model function trains the model.
- Data augmentation is applied during training.

## Evaluation
- The model is evaluated on a test dataset.
- Classification report and confusion matrix are generated.

## Prediction
- Functions for preprocessing images and making predictions are provided.
- Sample images are predicted and visualized.

## Visualization
- Functions to plot accuracy, loss, and confusion matrix are included.
  
  <img src="https://github.com/ArjunPramod/FireDetectNet/blob/main/images/Confusion%20Matrix.png" alt="Confusion Matrix" width="400" height="300">
  <img src="https://github.com/ArjunPramod/FireDetectNet/blob/main/images/Train%20vs%20Test%20loss%20plot.png" alt="Train vs Test loss plot" width="400" height="300">

## Results
- Training history, evaluation results, and sample predictions are presented.

  <img src="https://github.com/ArjunPramod/FireDetectNet/blob/main/images/Fire_prediction.png" alt="Predicted Fire plot" width="400" height="300">
  <img src="https://github.com/ArjunPramod/FireDetectNet/blob/main/images/noFire_prediction.png" alt="Predicted Non-Fire plot" width="400" height="300">
  
## Usage
1. **Training the Model:** Use the provided scripts to train the fire detection model on your dataset. Customize the model architecture and hyperparameters as needed.
2. **Evaluation:** Evaluate the model's performance using test data and visualize the results, including accuracy and a confusion matrix.
3. **Prediction:** Deploy the trained model to make predictions on new images for fire detection.

## License
This project is licensed under the Attribution-NonCommercial-NoDerivs 4.0 International (CC BY-NC-ND 4.0) [License](https://github.com/ArjunPramod/FireDetectNet/blob/main/LICENSE.md).

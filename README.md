# AI-Based Tool for Periodontitis Detection using CNN

## Project Overview

This project implements an AI-based tool using a Convolutional Neural Network (CNN) to detect periodontitis from dental images. The tool classifies the images into different stages of periodontitis, helping in early diagnosis and treatment planning for dental professionals. The project aims to improve accuracy and efficiency in dental image classification for automated diagnosis.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Model and Algorithms](#model-and-algorithms)
- [Results](#Results)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation

1. Clone the repository:
   <br/>
   ```git clone https://github.com/arthisri14/AI-based-tool-for-Periodontitis-detection-using-CNN.git```
   ```cd AI-based-tool-for-Periodontitis-detection-using-CNN```
   <br/>

2. Create a virtual environment and activate it:
   <br/>
   ```python -m venv venv```
   <br/>
   ```source venv/bin/activate```
   <br/>
   ```On Windows use `venv\Scripts\activate` ```

4. Install the required packages:
   <br/>
   ```pip install -r requirements.txt```

## Usage

1. Run the model:
   <br/>
   ```python predict.py --image path/to/image.jpg```
2. The model will output the classification result, indicating whether the image shows Normal, Gingivitis, or Periodontitis.

## Datasets

The dataset consists of labeled dental X-ray images used to classify different stages of periodontitis. It is divided into three main classes:

1.Normal
2.Gingivitis
3.Periodontitis

## Models and Algorithms

The tool utilizes a CNN model based on ResNet50, which is well-suited for image classification tasks. The key layers of the model include:

Input Layer: Takes in 224x224 pixel images.
Convolutional Layers: Several convolutional layers with filters of increasing depth to capture image features.
Pooling Layers: Max-pooling layers to reduce the spatial dimensions.
Fully Connected Layer: A dense layer for classification.
Output Layer: Softmax activation function for multiclass classification.

## Hyperparameters:
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Categorical Crossentropy
Batch Size: 32
Epochs: 50
Training
The model was trained using 80% of the dataset, with 20% reserved for validation. Key features of the training process include:

## Callbacks:
EarlyStopping: To stop training when the validation loss does not improve.
ReduceLROnPlateau: To reduce the learning rate if the model performance plateaus.

## Results

Test Accuracy:99%
Validation Accuracy: 99%
Loss: 0.001
The model achieved a high classification accuracy for the detection of periodontitis stages. Below are some key performance metrics:

Precision: 99%
Recall: 99%
Confusion matrices and accuracy/loss curves are available in the project folder for better performance visualization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the open-source community for providing the datasets and tools necessary for this project.

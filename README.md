# Knee X-ray Image Classification for Osteopenia / Osteoporosis Detection
This repository contains code and resources for a project focused on investigating the effectiveness of image classification models for identifying osteopenia and osteoporosis in knee X-ray images. The goal is to develop a reliable and accurate model that can assist in diagnosing these conditions from X-ray images.
## Dataset
The 'dataset2' folder contains a collection of 744 single knee X-ray images taken from Kaggle, with an equal distribution between healthy patients and those diagnosed with osteoporosis
## Model 1: EfficientNet-B0
The 'Efficientnet-b0.ipynb' notebook implements the efficientnet-b0 model (Mingxing Tan, Quoc V. Le 2020) using k-fold cross-validation (k=5) on the aformentioned dataset, achieving a mean accuracy of 95%. In addition, a demo single image classifier using a trained model has been added along with a Grad-CAM overlay.

## Model 2: YoloV5
The 'yolov5.ipynb' notebook trains a yolov5 model using the dataset through random split

## Currently under development...

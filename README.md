# Knee X-ray Image Classification for Osteopenia / Osteoporosis Detection
This repository contains code and resources for a project focused on investigating the effectiveness of image classification models for identifying osteopenia and osteoporosis in knee X-ray images. The goal is to develop a reliable and accurate model that can assist in diagnosing these conditions from X-ray images.
## Dataset
The 'dataset2' folder contains a collection of 744 single knee X-ray images taken from Kaggle, with an equal distribution between healthy patients and those diagnosed with osteoporosis
## Model 1: EfficientNet-B0
The 'Efficientnet-b0.ipynb' notebook implements the efficientnet-b0 model (Tan, M. & Le, Q.V. 2019) using k-fold cross-validation (k=5) on the aformentioned dataset, achieving a mean accuracy of 95%. In addition, a demo single image classifier using a trained model has been added along with a Grad-CAM overlay.

## Model 2: YoloV5
The 'yolov5.ipynb' notebook trains a YOLOv5 model (Jocher, G. et al. 2022) using the dataset through random split using Roboflow. Validation confirms the model has an accuracy of 82%.

## App deployment
The files in the 'idrc' and 'Swift App' folders contains an Apple app interface using Swift which deploys the trained efficientnet-b0 model along with a Grad-CAM visualization onto any uploaded image. In addition, this application can directly interface with Apple Vision Pro.

## Contributors
Freddie Nicholson<br />
James Pang<br />
Michael Fang<br />
Amandeep Kaur

## References
Tan, M. and Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.11946.<br />
Jocher, G., Chaurasia, A., Stoken, A., Borovec, J., NanoCode012, Kwon, Y., Michael, K., TaoXie, Fang, J., Imyhxy, Lorna, 曾逸夫Z.Y., Wong, C., V, A., Montes, D., Wang, Z., Fati, C., Nadar, J., Laughing and UnglvKitDe (2022). ultralytics/yolov5: v7.0 - YOLOv5 SOTA Realtime Instance Segmentation. Zenodo. [online] doi:https://doi.org/10.5281/zenodo.7347926.

‌

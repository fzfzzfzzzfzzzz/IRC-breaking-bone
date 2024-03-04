
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

train_transforms = transforms.Compose([
    transforms.Resize((256,256)), # Make sure this is the same size as used for calculating mean and std,
    transforms.RandomHorizontalFlip(), # Randomise direction of image,
    transforms.RandomRotation(10),
    #transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95,1.05)), 
    transforms.ToTensor(), # Multidimensional array,
])

test_transforms = transforms.Compose([
    transforms.Resize((256,256)), # Make sure this is the same size as used for calculating mean and std,
    transforms.ToTensor(), # Multidimensional array,
])

checkpoint = torch.load('model_best_checkpoint.pth.tar', map_location=torch.device('cpu'))
efficientnet_model = models.efficientnet_b0(pretrained=False)
num_classes = 2
efficientnet_model.fc = nn.Linear(1280, num_classes)
efficientnet_model.load_state_dict(checkpoint['model'])
#torch.save(efficientnet_model, 'best_model.pth')

import PIL.Image as Image
def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(f'Prediction: {classes[predicted.item()]}')

classes = ['Normal', 'Osteoporosis']
classify(efficientnet_model, test_transforms, 'normal_example.jpg', classes)
classify(efficientnet_model, test_transforms, 'osteoporosis_example.png', classes)
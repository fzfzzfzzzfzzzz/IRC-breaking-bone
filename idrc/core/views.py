from django.shortcuts import render,redirect
from .forms import ImageUploadForm
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
import base64

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
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
            model = efficientnet_model
            image_transforms = test_transforms
            classes = ['Normal', 'Osteoporosis']
            model = model.eval()
            image = Image.open(image_file)
            image = image_transforms(image).float()
            image = image.unsqueeze(0)

            output = model(image)
            _, predicted = torch.max(output.data, 1)

            classifier = classes[predicted.item()]
            
            return render(request, 'index.html', {'form': form, 'img': image_data, 'classifier': classifier})
    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})
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
from torchcam.methods import SmoothGradCAMpp, SSCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from matplotlib.animation import FuncAnimation
from io import BytesIO
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from io import BytesIO
from django.http import JsonResponse

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
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, sep = torch.max(probs, 1)
            conf = round(float(conf[0])*100, 3)
            print(f'Prediction: {classes[predicted.item()]} ({conf}%)')

            classifier = classes[predicted.item()]

            def process_gradcam(model, image_transforms, image_path, classifier, conf):
                device = 'cpu'
                mode = model.to(device)
                model = model.eval()

                image = Image.open(image_path)
                input_tensor = image_transforms(image)
                transform = transforms.ToTensor()
                output_tensor = transform(image)



                with SmoothGradCAMpp(model) as cam_extractor:
                    # Preprocess your data and feed it to the model
                    out = model(input_tensor.unsqueeze(0))
                    # Retrieve the CAM by passing the class index and the model output
                    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)



                # Resize the CAM and overlay it
                result = overlay_mask(to_pil_image(output_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
                buffer = BytesIO()
                result.save(buffer, format="JPEG")
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                # Display it
                # return HttpResponse("Hello")
                return render(request, 'index.html', {'form': form, 'img': image_data, 'classifier': classifier, 'conf': conf})

            return process_gradcam(model, image_transforms, image_file, classifier, conf)
        else:
            return HttpResponse("invalid")
            
    else:
        form = ImageUploadForm()
        return render(request, 'index.html', {'form': form})

@csrf_exempt
def ipad(request):
    if request.method == 'POST':
        alpha_level = request.GET['alpha']
        alpha_level = int(round(float(alpha_level)))/100
        # form = ImageUploadForm(request.POST, request.FILES)
        json_data = json.loads(request.body)
        # image_file = request.FILES['image']
        image_data = json_data["image"]
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
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.convert('RGB')
        image = image_transforms(image).float()
        image = image.unsqueeze(0)

        output = model(image)
        _, predicted = torch.max(output.data, 1)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, sep = torch.max(probs, 1)
        conf = round(float(conf[0])*100, 3)
        print(f'Prediction: {classes[predicted.item()]} ({conf}%)')

        classifier = classes[predicted.item()]

        def process_gradcam(model, image_transforms, image_path, classifier, conf, alpha_level):
            device = 'cpu'
            mode = model.to(device)
            model = model.eval()

            image = image_path
            image = image.convert('RGB')
            input_tensor = image_transforms(image)
            transform = transforms.ToTensor()
            output_tensor = transform(image)



            with SmoothGradCAMpp(model) as cam_extractor:
                # Preprocess your data and feed it to the model
                out = model(input_tensor.unsqueeze(0))
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)



            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(output_tensor), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=alpha_level)
            buffer = BytesIO()
            result.save(buffer, format="JPEG")
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            # Display it
            # return HttpResponse("Hello")
            return JsonResponse({ 'img': image_data, 'classifier': classifier, 'conf': conf})

        return process_gradcam(model, image_transforms, Image.open(BytesIO(base64.b64decode(image_data))), classifier,conf, alpha_level)

            
    else:
        form = ImageUploadForm()
        return render(request, 'index.html', {'form': form})
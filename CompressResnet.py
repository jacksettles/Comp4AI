#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Torch stuff
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.optim as optim

# Compression modules
import numpy as np
from pathlib import Path
from Comp4AI.notebooks.pysz.pysz import SZ
import zfpy
import matplotlib.pyplot as plt

import sys
import time
import argparse


# In[19]:


if torch.cuda.is_available():
    # CUDA is available, you can proceed to use it
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    # CUDA is not available, use CPU
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


# In[20]:


resnet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Resize((256,256))])


# In[21]:


pattern = 'subclasses-tiny-imagenet/*/images/*.JPEG'
jpeg_files = glob.glob(pattern, recursive=True)


# In[22]:


class FineTuningDataset(Dataset):
    def __init__(self, data, transform = resnet_transform):
        '''class_dict = {'n02410509':347,
                      'n02106662':235,
                      'n07734744':947,
                      'n07873807':963,
                      'n07920052':967,
                      'n09428293':978,
                      'n01910747':107,
                      'n01882714':105,
                      'n04285008':817,
                      'n04146614':779}'''
        
        class_dict = {'n02410509':0,
                      'n02106662':1,
                      'n07734744':2,
                      'n07873807':3,
                      'n07920052':4,
                      'n09428293':5,
                      'n01910747':6,
                      'n01882714':7,
                      'n04285008':8,
                      'n04146614':9}
        
        data_and_labels = []
        for i in range(len(data)):
            file_path = data[i]
        
            # Split the file string
            split_file = data[i].split('/')
            class_wnid = split_file[1]
            file_name = split_file[3]
        
            with Image.open(file_path) as img:
            
                # Convert grayscale images to RGB
                if img.mode == 'L':
                    img = img.convert('RGB')
            
                img_tensor = resnet_transform(img)
            
            data_and_labels.append((img_tensor, class_dict[class_wnid]))
        
        self.data = data_and_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx][0]
        target = self.data[idx][1]
        
        return sample, target


# In[23]:


def fine_tune(model=None, train_loader=None, loss_func=None, model_name=None, num_epochs=5, optimizer=None, test_loader=None):
    model = model.to(device)
    highest_acc = 0
    
    for epoch in range(1, num_epochs+1):
        train_correct = 0
        train_total = 0
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            model.train()   # set the model in training mode
                
            images = images.to(device) # move images and labels to GPU
            labels = labels.to(device)
                
            optimizer.zero_grad() # Zero out gradients from last backprop
                
            outputs = model(images) # Pass images through the model
            _, predicted = torch.max(outputs.data, 1) # Obtain indices of predictions
                
            train_loss = loss_func(outputs, labels) # Get the loss and backpropogate it
            train_loss.backward()
                
            # Get some metrics
            running_loss += train_loss
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()
            train_accuracy = 100 * (train_correct/train_total)
                
            if step % 5 == 0:
                test_acc, test_loss = test(model=model, test_loader=test_loader, loss_func=loss_func)
                print(f"Epoch: {epoch}, Step: {step}")
                print(f"\tTrain Loss: {running_loss}, Train Accuracy: {train_accuracy}")
                print(f"\tTest Loss: {test_loss}, Test Accuracy: {test_acc}")
                print()
                if test_acc > highest_acc:
                    highest_acc = test_acc
                    torch.save(model.state_dict(), f'{model_name}.pt')
                
            optimizer.step()


# In[24]:


def test(model=None, test_loader=None, loss_func=None):
    total = 0
    correct = 0
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in test_loader:            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            test_loss = loss_func(outputs, labels)
            total_loss += test_loss
            
            # Increment total number of observations seen by
            # number of items in this batch
            total += labels.size(0)

            # Increment total number of correct predictions by
            # number of correct predictions in this batch
            correct += (predicted == labels).sum()
            
        accuracy = (100 * (correct/total))
        accuracy = accuracy.item()
        return accuracy, total_loss


# In[25]:


def print_memory_usage(stage):
    print(f"{stage} - Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


# In[26]:


def compress_intermediate_results(model=None, 
                                 mode="train", layer=0, tolerance=1e-3,
                                 model_name=None, criterion=None, optimizer=None,
                                 train_loader=None, test_loader=None, num_epochs=5):
    def zfpy_compress_output(module, input, output):
        if mode != "train":
            output = output.cpu().numpy() # For testing
        else:
            output = output.cpu().detach().numpy() # For training
        compressed_data = zfpy.compress_numpy(output, tolerance=tolerance)
    #     compressed_data = zfpy.compress_numpy(output, rate=8)
        decompressed_array = zfpy.decompress_numpy(compressed_data)
        output_dec = torch.from_numpy(decompressed_array).to(device)
        return output_dec

    if layer == 0:
        hook = model.maxpool.register_forward_hook(zfpy_compress_output)
    elif layer == 1:
        hook = model.layer1[-1].register_forward_hook(zfpy_compress_output)
    elif layer == 2:
        hook = model.layer2[-1].register_forward_hook(zfpy_compress_output)
    elif layer == 3:
        hook = model.layer3[-1].register_forward_hook(zfpy_compress_output)
    elif layer == 4:
        hook = model.layer4[-1].register_forward_hook(zfpy_compress_output)

    start_time = time.perf_counter()
    
    if mode == "train":
        fine_tune(model=model, train_loader=train_loader, loss_func=criterion, model_name=model_name, optimizer=optimizer,
                 test_loader=test_loader, num_epochs=num_epochs)
    else:
        accuracy, test_loss = test(model=model, test_loader=test_loader, loss_func=criterion)
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    hook.remove()
    
    if mode == "train":
        print()
        print("-------------------")
        print("Finished training!!")
        print("Layer: {}, Tolerance: {}, Total training time: {}:{}".format(layer, tolerance, int(minutes), int(seconds)))
        print("-------------------")
        return tolerance, int(minutes), int(seconds)
    else:
        print("Accuracy with zfpy: {:.2f}%\tTime spent: {}:{}".format(accuracy, int(minutes), int(seconds)))
        return accuracy, int(minutes), int(seconds)


# In[43]:


def plot_training(tolerances, training_times, layer):
    
    tolerances_str = [str(tol) for tol in tolerances]
    
    plt.figure(figsize=(10, 6))
    plt.bar(tolerances_str, training_times, width=0.05, log=True)
    
    plt.xlabel('Tolerance Levels')
    plt.ylabel('Training Time (minutes)')
    
    # Adding a title
    plt.title(f'Training Time vs Tolerance Levels for layer {layer}')

    # Set x-axis to log scale for better readability
    plt.xticks(tolerances_str)

    plt.savefig(f'comp_training_images/layer{layer}_training_time_vs_tolerance.png', dpi=300, bbox_inches='tight')
    
    # Display the chart
    plt.show()


# In[44]:


def plot_testing(tolerances, accuracies, layer):
    
    tolerances_str = [str(tol) for tol in tolerances]
    
    plt.figure(figsize=(10, 6))
    plt.bar(tolerances_str, accuracies, width=0.05, log=True)
    
    plt.xlabel('Tolerance Levels')
    plt.ylabel('Accuracy')
    
    # Adding a title
    plt.title(f'Accuracy vs Tolerance Levels during training for layer: {layer}')

    # Set x-axis to log scale for better readability
    plt.xticks(tolerances_str)

    plt.savefig(f'comp_training_images/layer{layer}_accuracy_vs_tolerance.png', dpi=300, bbox_inches='tight')


# In[31]:


parser = argparse.ArgumentParser()

parser.add_argument('--mode', default="train", choices=["train", "test"], help="Compress during training or testing")
parser.add_argument('--layer', default=0, choices=[0, 1, 2, 3, 4], help="Which stage of the model do you want to compress the intermediate results from")
parser.add_argument('--model_name', choices=['zfpy', 'sz3'], help='name you want to save the model to')
parser.add_argument('--tolerance', default=1e-3, choices=[1, 1e-1, 1e-2, 1e-3, 1e-4], help="Tolerance you want to pass through to the compression algorithm")


# In[32]:


num_epochs = 5
batch_size = 128

data = FineTuningDataset(jpeg_files)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[36]:


args = parser.parse_args()
layers = [0, 1, 2, 3, 4]
tols = [1, 1e-1, 1e-2, 1e-3]

for layer in layers:
    times = []
    accuracies = []
    for tol in tols:
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_classes = 10
        resnet.fc = nn.Linear(2048, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

        model_name = args.model_name
        model_name = "models/" + model_name + "_layer_" + str(layer) + "_tolerance_" + str(tol)

        _, minutes, seconds = compress_intermediate_results(model=resnet,
                                                            mode=args.mode,
                                                            layer=layer,
                                                            tolerance=tol,
                                                            model_name=model_name,
                                                            criterion=criterion,
                                                            optimizer=optimizer,
                                                            train_loader=train_loader,
                                                            test_loader=test_loader,
                                                            num_epochs=num_epochs)
        times.append(minutes)

        # Load the model you just saved to test it now
        resnet.load_state_dict(torch.load(f'{model_name}.pt'))

        accuracy, test_loss = test(model=resnet, test_loader=test_loader, loss_func=criterion)
        accuracies.append(accuracy)
    
    plot_training(tols, times, layer)
    plot_testing(tols, accuracies, layer)



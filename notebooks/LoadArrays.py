#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
from typing import List
import glob
import matplotlib.pyplot as plt
import math
import torch


# In[92]:


def load_data(input_data_stage: int, class_name: str=None) -> List[np.ndarray]:
    """
    This function loads in the feature map data from a pretrained ResNet50 model.
    If you specify 1, this will load the data that is used as input for stage 1 of the model.
    If you specify 2, this will load the data that is used as input for stage 2 of the model.
    
    Stage 1 numpy arrays are returned with a shape of 64x64x64 (64 feature maps with HxW 64x64).
    Stage 2 numpy arrays are returned with a shape of 256x64x64 (256 feature maps with HxW 64x64).
    Call this function and store it in a variable, say 'arr'. Calling 'arr[0]' will return the first
    set of feature maps for a specific image from tiny-imagenet-200.
    
    :param input_data_stage: This must be an integer value of either 1 or 2.
        This specifies which stage of the ResNet50 model that you want to load the data for.
        
    :param class_name: This is optional. If you wish to only load images of a certain class,
        then pass the string value of this class into the function. Check the load_class_names function
        to see what class names there are, and what literal you must pass.
        
    :return: A List object of numpy arrays.
    """
    
    pattern = 'resnet50/'
    stage_dir = ""
    
    if input_data_stage == 1:
        stage_dir = 'cfg0_input_data/'
    elif input_data_stage == 2:
        stage_dir = 'cfg1_input_data/'
        
    if class_name != None:
        classes = load_class_dict()
        class_path = classes[class_name]
    else:
        class_path = '*'
        
    pattern = pattern + stage_dir + class_path + '/*.npy'
        
    jpeg_files = glob.glob(pattern, recursive=True)
        
    data_arrays = []
    for file in jpeg_files:
        array = np.load(file)
        data_arrays.append(array)
        
    return data_arrays


# In[93]:


def plot_img(feature_maps):
    """
    If you wish to visualize the feature maps using matplotlib, pass one into this function.
    If you called
            arr = load_data(1)
    above, then arr is a list of numpy arrays. arr[0] is the first set of feature maps in that list. 
    These feature maps correspond to a specific image in the tiny-imagenet-200 dataset.
    You can pass arr[i] for any i into this function to plot the feature maps.
    
    :param feature_maps: A numpy array of feature maps
    """
    # Turn numpy array into a torch tensor
    feature_maps = torch.from_numpy(feature_maps)
    
    num_filters = feature_maps.size(0)
    dim = int(math.sqrt(num_filters))

    # Visualize the feature maps
    
    fig, axarr = plt.subplots(dim, dim, figsize=(15, 15))  # Assuming 32 filters, adjust as needed
    for i in range(num_filters):
        ax = axarr[i // dim, i % dim]

        image_array = feature_maps[i].detach().cpu().numpy() # Turn back into numpy array

        ax.imshow(image_array, cmap='viridis')
        ax.axis('off')
    plt.show()


# In[94]:


def load_class_dict():
    """
    Returns a dictionary. Keys are string literals, values are the WordNet IDs associated with these classes.
    These are also the names of the directories within the data.
    """
    class_dict = {'bison': 'n02410509',
           'german shepherd': 'n02106662',
           'mushroom': 'n07734744',
           'pizza': 'n07873807',
           'espresso': 'n07920052',
           'seashore': 'n09428293',
           'jellyfish': 'n01910747',
           'koala': 'n01882714',
           'sports car': 'n04285008',
           'school bus': 'n04146614'}
    return class_dict


# In[95]:


def load_class_names():
    '''
    Returns a list of class names in the dataset.
    These classes were simply handpicked from the classes used in tiny-imagenet-200.
    '''
    class_dict = load_class_dict()
    class_names = list(class_dict.keys())
    return class_names


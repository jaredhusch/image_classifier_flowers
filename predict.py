'''
This programs predicts a flower name from an image with predict.py along with the probability of that name.
Pass in a single image and return the flower name and class probability.

    Basic usage: python predict.py
    Options:
        Load checkpoin: python predict.py --checkpoint
        Loading the image to be predicted: python predict.py --input_img
        Return the top most likekly classes: python predict.py --top_k
        Use a mapping of categories to real names: python predict.py input --category_names
        Use GPU for inference: python predict.py --gpu
'''


import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import json
import utils
import argparse
import train
import utils


# Main program function defined below
def main():
    
    archs = {"vgg16": 25088,
             "densenet121": 1024,
             "alexnet" : 9216 }

    print('Hello and Welcome!')
    #read parameters
    arguments = read_args()

    path = arguments.checkpoint
    input_img = arguments.input_img
    top_k=arguments.top_k
    category_names=arguments.category_names

    force_processor = 'cuda'
    processor = ('cuda' if torch.cuda.is_available() and force_processor == 'cuda' else 'cpu')
    #generate and normalize images for testing, training and validation
    train_loader, test_loader, valid_loader = load_data()
    #load checkpoint
    model, model_class = load_checkpoint(path)

    with open('cat_to_name.json', 'r') as json_file:
      cat_to_name = json.load(json_file)
    #predict classification 
    flower_probs, flower_classes = predict(input_img, model,top_k)
    
    #chew stats and print results 
    cat_to_name = load_cat_names(category_names)
    print_results(flower_probs,
                  flower_classes,
                  model_class,
                  input_img,
                  cat_to_name)

    print("**INFO: Prediction Ended! Send another!")



def load_cat_names(cat_to_name):
    with open(cat_to_name, 'r') as f:
        categories = json.load(f)
        return categories

def read_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: path to load.
    parser.add_argument('--checkpoint', type = str, default="./checkpoint_flower.pth",
                        help = 'path to the saved trained model')

    # Argument 2: path to image file to check. 
    parser.add_argument('--input_img', type = str, default="./flower_data/test/99/image_07874.jpg",
                        help = 'image path to be predicted')

    # Arguments 3: top k
    parser.add_argument('--top_k', type = int, default = '5',
                        help = 'display top KKK most likely classes')
    # Arguments 4: category names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'mapping of categories to real names')

    # Argument 5: choose power
    parser.add_argument('--gpu', action='store_true',
                        help = 'use gpu acceleration if available')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    print("Arguments: {} ", in_args)

    return in_args

def load_data( data_dir= "./flower_data/"):

  data_dir = data_dir
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
  test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
  valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
  train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
  test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
  valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

  trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
  validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

  return trainloader, validloader, testloader

def load_checkpoint(file_dir):  # Load pre-trained model checkpoint
    '''
        A function that loads a checkpoint and rebuilds the model
    '''
    # Load saved pre-trained model checkpoint
    checkpoint = torch.load(file_dir, map_location='cpu')

    # Get model architecture
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Model arch not found.")

    # Feature parameters
    for x in model.parameters():
        x.requires_grad = False

    # Load model parameters
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['img_mapping']

    return model, model.class_to_idx

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('**INFO: Start image prediction of '+image_path)
    img = utils.process_image(image_path)
    img = img.unsqueeze(0)

    model.type(torch.DoubleTensor)
    model.eval()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        probability = model.forward(img)

    probs, classes = torch.exp(probability).topk(topk)
    np_prob = np.asarray(probs)[0]
    np_classes = np.asarray(classes)[0]

    print('**INFO: Finished predition')
    return (np_prob, np_classes)

def get_folder_key(mydict,value):
    return list(mydict.keys())[list(mydict.values()).index(value)]

def get_class_value(categories_names, key):
    return categories_names[key]  

def print_results(flower_probs, flower_classes, img_mapping, img_path, categories_names):
    flower_names = []
    for i in range(len(flower_probs)):
        most_prob = str(round(flower_probs.item(i)*100, 4))
        most_prob_flower_id = flower_classes.item(i)
        most_prob_flower_id_folder = get_folder_key(img_mapping,most_prob_flower_id)
        most_prob_flower = get_class_value(categories_names,most_prob_flower_id_folder)
        flower_names.append(most_prob_flower)
        print("The image has a {:.2f}% probability that is a {}".format(float(most_prob),most_prob_flower))

# Call to main function to run the program
if __name__ == '__main__':
    main()
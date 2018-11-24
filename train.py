'''
Train a network on a data set with train.py
    Default Usage: python train.py 
    Prints out the training loss, validation loss, and validation accuracy
    Options:
        Set directory to load images: python train.py --data_dir 
        Set directory to save checkpoints: python train.py --save_dir 
        Choose architecture: python train.py --arch "vgg16"  Note: There are three archs= ("vgg16","densenet121","alexnet")]
        Set hyperparameters: python train.py --learning_rate 0.001 --hidden_units 1000 --epochs 7
        Use GPU for training: python train.py --gpu
'''


# Imports
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


# Main program function defined below
def main():
    dropout_per = 0.25
    n_classes = 102
    archs = {"vgg16": 25088,
             "densenet121": 1024,
             "alexnet" : 9216 }
    data_dir= 'flowers'
    steps = 0
    print_every = 10
    print('Hello and Welcome!')
    #read parameters
    arguments = read_args()
    arch = arguments.arch
    lr = arguments.learning_rate
    hidden_layer1 = arguments.hidden_units
    epochs = arguments.epochs

#generate and normalize images for training, testing, and validation
    trainloader,validloader,testloader,train_data, valid_data, test_data = load_data(arguments.data_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

 #download pre-trained model and initialize
    model, optimizer, criterion, classifer = nn_setup(archs, arch,dropout_per,hidden_layer1,lr)

    force_processor = 'cuda'
    processor = ('cuda' if torch.cuda.is_available() and force_processor == 'cuda' else 'cpu')
    model.to(processor)

#train network

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(processor), labels.to(processor)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, processor)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                # Make sure training is back on
                running_loss = 0
                model.train()
                
    print('finished training')
    
    #Check Accuracy on Test
    model.eval()
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for data in testloader:
            count += 1
            images, labels = data
            if processor == 'cuda':
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {}%'.format(count, (100 * correct / total)))

	#save checkpoint

    save_params_dict = {'hidden_layer1': hidden_layer1,
                        'output_size': n_classes,
                        'learning_rate': lr,
                        'dropout_per': dropout_per,
                        'epochs_trained': epochs,
                        'inputs': archs[arch],
                        'classifier': model.classifier,
                        'arch': arguments.arch, 
                        'img_mapping': train_data.class_to_idx,
                        'optimizer_state': optimizer.state_dict(),
                        'model_state': model.state_dict()}
                        
    torch.save(save_params_dict, 'checkpoint_flower.pth')
     
    print('Done!')

# Implement a function for the validation pass
def validation(model, testloader, criterion, processor):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images = images.to(processor)
        labels = labels.to(processor)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy    

def nn_setup(archs, arch,dropout, hidden_layer1,lr):
  
  if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
  elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
  elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
  else:
    print("Not a valid model!")

  for param in model.parameters():
      param.requires_grad = False

  classifier = nn.Sequential(OrderedDict([
          ('dropout',nn.Dropout(dropout)),
          ('inputs', nn.Linear(archs[arch], hidden_layer1)),
          ('relu1', nn.ReLU()),
          ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
          ('relu2',nn.ReLU()),
          ('hidden_layer2',nn.Linear(90,80)),
          ('relu3',nn.ReLU()),
          ('hidden_layer3',nn.Linear(80,102)),
          ('output', nn.LogSoftmax(dim=1)) 
          ]))

  model.classifier = classifier

  optimizer = nn.NLLLoss()
  criterion = optim.Adam(model.classifier.parameters(), lr)
  return model, criterion, optimizer, classifier    

def load_data(data_dir):

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

  return trainloader, validloader, testloader, train_data, valid_data, test_data

def read_args():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 0: data directory (flower_data)
    parser.add_argument('--data_dir', type = str, default="./flower_data/",
                        help = 'name of the subfolder that contains the datasets')
    
    # Argument 1: path to save checkpoints
    parser.add_argument('--save_dir', type = str, default = './',
                        help = 'path to save the trained model')

    # Argument 2: choose archietcture
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'name of the chosen architecture')

    # Arguments 3-5: training hyperparameters
    parser.add_argument('--learning_rate', type = float, default = '0.001',
                        help = 'training: learning rate')

    parser.add_argument('--hidden_units', type = int, default = '1000',
                        help = 'training: hidden units')

    parser.add_argument('--epochs', type = int, default = '7',
                        help = 'training: number of epochs')

    # Argument 6: choose power
    parser.add_argument('--gpu', action='store_true',
                        help = 'use gpu acceleration if available')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()
    print("Arguments: {} ", in_args)

    return in_args

# Call main function that runs the program
if __name__ == '__main__':
    main()
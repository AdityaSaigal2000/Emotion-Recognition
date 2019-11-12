import cv2
import os
import detectFaces
import rpnModel



import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import glob




def load_images(folder):
    images = []
    for i in range(0,30):
      img = mpimg.imread(folder + 'outputImage'+str(i)+'.jpg')
      if img is not None:
          images.append(img)
    return images


path =os.getcwd()

imageName = "outputImage11.jpg"
outputImagesFolder = "\\outputImages\\"


imagePath = path+outputImagesFolder+ imageName


#############
#the cv2 image is not necessary in this program
image = cv2.imread(imagePath, cv2.IMREAD_COLOR) 

#cv2.imshow("", image)
#cv2.waitKey()
#cv2.destroyAllWindows()
#############

#get the images in tensor form
images = load_images(path+outputImagesFolder)
#print(len(images))

#show the images
'''
fig = plt.figure(figsize=(25, 4))
for idx in range(0,len(images)):
  ax = fig.add_subplot(2, len(images)/2, idx+1, xticks=[], yticks=[])
  plt.imshow(images[idx])
plt.show()
'''

# define dataloader parameters
batch_size  = 1
num_workers = 0
# prepare data loaders
img = torch.FloatTensor(images)
print(img)
train_loader = torch.utils.data.DataLoader(img, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=False)

#make the rpn model objects
RPNFeatureExtractor = rpnModel.RPNFeatureExtractor()
#RPNFeatures = RPNFeatureExtractor(image)
RPN = rpnModel.RPNmodel()

#see the output of the 4image in the rpn
counter = 0
for imgs in train_loader:
  if counter == 0:
    counter+= 1
    continue
  imgs= imgs.permute(0,3,1,2)
  print("input image shape:" + str(imgs[0].shape))
  featuresOut = RPNFeatureExtractor(imgs)
  print("features out shape: " + str(featuresOut.shape))
  features = featuresOut.detach().numpy()
  features = (features - features.min()) / (features.max() - features.min())
  
  plt.figure(figsize=(10,10))
  for i in range(64):
    plt.subplot(8, 8, i+1)
    plt.imshow(features[0, i])
  plt.show()
  
  outputROIs = RPN(featuresOut)
  print(outputROIs.shape)
  print(outputROIs)
  #features = alexNet.features(imgs)
  #print(features.shape)


import numpy as np
import cv2
import os
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import sklearn
#import sklearn.cross_validation
import sklearn.model_selection
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import random
import pandas as pd


path =os.getcwd()
#imageFolder = "\\imagesIn\\"
imageOutputFolder = "\\outputImages\\"


imageFolder = "\\faceImages\\"
#imageOutputFolder = "\\faceCuts\\"
imageName = "faceReference.jpg"
datasetFolder = "\\EmotionPics\\"
backgroundFolder = "\\backgrounds\\"
backgroundImage = "background1.jpg"
outputFaceDataTextFile = "faceData.txt" 
kaggleDatasetFile = "fer2013.csv" #THIS FILE MUST BE ADDED TO THE FOLDER

imagePath = path+imageFolder+ imageName
outputPath = path + imageOutputFolder
datasetPath = path + datasetFolder
backgroundPath = path + backgroundFolder + backgroundImage
kagglePath  = path + "\\" + kaggleDatasetFile

imageName = "1_I_1.jpg"


numRows = 3
numCols = 3
numOutputImages = 30
faceWidth = 70
#############

imagePath = path+imageFolder+ imageName
outputPath = path + imageOutputFolder



#print (imagePath)
#image = cv2.imread(imagePath, cv2.IMREAD_COLOR) 
#######################################################
def get_data_loader():
    data = pd.read_csv(kaggleDatasetFile)  
    images = data['pixels'].tolist()
    input_data = []
    for image in images:
        temp = [int(pixel) for pixel in image.split(' ')]
        temp = np.asarray(temp).reshape(48, 48)
        input_data.append(temp.astype('float32'))
    input_data = np.asarray(input_data)
    input_data = np.expand_dims(input_data, 1)
#    target = pd.get_dummies(data['emotion']).values
    target = data['emotion'].tolist()
    
    tensor_data = torch.stack([torch.Tensor(i) for i in input_data])
#    tensor_target = torch.stack([torch.Tensor(i) for i in target])
    tensor_target = torch.LongTensor(target)
    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_target)

    # Get the list of indices to sample from
    indices = np.array(range(len(dataset)))
    split1 = 28709
    split2 = 32298
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    batch_size = len(train_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=0, sampler=train_sampler)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    batch_size = len(val_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=val_sampler)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    batch_size = len(test_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader
    

#take inputs from the kaggle data file

############################
#get data

# define training and test data directories

'''
#this is old and was for the test pictures... before kaggle dataset was being used
# classes are folders in each directory with these names
classes = ['frown', 'neutral', 'smile']

# resize all images to 224 x 224
data_transform = transforms.Compose([transforms.Resize((faceWidth,faceWidth)), transforms.ToTensor()])

data = datasets.ImageFolder(datasetPath, transform=data_transform)
# print out some data stats
print('Num images: ', len(data))

# prepare data loaders
batch_size = len(data)
num_workers = 0
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                        num_workers=num_workers, shuffle=True)
# Visualize some sample data
'''
#Now that kaggle dataset is being used call the 
train_loader, val_loader, test_loader = get_data_loader()
classes = ['angry', 'disgust', 'fear', 'sad', 'surprise', 'neutral']
# obtain one batch of training images
dataiter = iter(train_loader)
imagesRaw, labelsRaw = dataiter.next()
imagesnp = imagesRaw.numpy() # convert images to numpy for display
labelsnp = labelsRaw.numpy()

# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(9):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(imagesnp[idx], (1, 2, 0)))
    ax.set_title(classes[labelsnp[idx]])
plt.show()

#for i in range(len(imagesnp)):
#    imagesnp[i] = np.transpose(imagesnp[i], (1, 2, 0))
'''
test_image = np.transpose(imagesnp[3], (1, 2, 0))
RGB_img = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
cv2.imshow('Main',RGB_img)
cv2.waitKey()
'''
imagesnpTrans = np.transpose(imagesnp, (0, 2, 3, 1))
#print(classes[labelsnp[3]])
imagesnpTransList = []
for i in imagesnpTrans:
    #i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)                                                                                                                                                                                                                         )
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    imagesnpTransList += [np.uint8(np.interp(i, (i.min(), i.max()), (0,255)))]
#imagesnpTransList = np.asarray(imagesnpList)

#Images_train, Images_test, Labels_train, Labels_test = train_test_split(imagesnpList, labelsnp, test_size = 0.25, random_state = 0)
#return Images_train, Images_test, Labels_train, Labels_test

#FINAL OUTPUTS:
#labelsnp
#imagesnpTransList
##############################


#cv2.imshow("",image)

#cv2.waitKey()

# I just resized the image to a quarter of its original size
#image = cv2.resize(image, (0, 0), None, .25, .25)


backgroundImage = cv2.imread(backgroundPath, cv2.IMREAD_COLOR) 
backgroundImage = cv2.resize(backgroundImage, (faceWidth, faceWidth))
'''
cv2.imshow("", backgroundImage)
cv2.waitKey()
cv2.destroyAllWindows()
print(print((backgroundImage[0][0][0])))
'''
#backgroundImage = backgroundImage.astype(np.float32)/255.0
#backgroundImage = backgroundImage.astype(np.float32)/255.0

'''
cv2.imshow("", backgroundImage.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
print(backgroundImage)
print(print((backgroundImage[0][0][0])))


cv2.imshow("", imagesnpTransList[0])
cv2.waitKey()
cv2.destroyAllWindows()
print((imagesnpTransList[0][0][0][0]))
print(imagesnpTransList[0])
'''

outputFaceData = []
for outputNumber in range(numOutputImages):

    images = []
    lenRandomNumbers = len(imagesnpTransList)
    labelsOutput = [] #[1, 0, 2 etc....]
    positionsOutput = [] #[[y1, x1, y2, x2], [etc...]]

    #make a 2D list with random images, and another 2d list with the corresponding labels, and another list with positions
    for row in range(numRows):    
        imageRow = []
        labelRow = []
        for col in range(numCols):
            #randomly choose whether to put image or not:
            if (random.randint(0,10) < 2):
                index = random.randint(0,lenRandomNumbers-1)
                imageRow += [imagesnpTransList[index]]
                labelRow += [labelsnp[index]]
                #add the face label and location data to the 
                x1 = faceWidth*col
                y1 = faceWidth*row
                x2 = faceWidth*col+(faceWidth-1)
                y2 = faceWidth*row+(faceWidth-1)
                
                outputFaceData += [str(outputNumber) + "," + str(labelsnp[index]) + "," + str(x1)+ "," + str(y1)+ "," + str(x2)+ "," + str(y2)]
                
            else:
                #if not putting an image still increase the length of the arrays
                #also put dummy image
                imageRow += [backgroundImage]
                labelRow += []#-1 can be used to represent the background image
        images+=[imageRow]
        labelsOutput += [labelRow]


    combinedImage = None
    for row in range(numRows):
        currentRowImage = images[row][0]
        for col in range(1,numCols):
            #print(images[row][col].shape)
            currentRowImage = np.concatenate((currentRowImage, images[row][col]), axis=1)
            #print((images[row][col][0][0][0]))
            #cv2.imshow("", currentRowImage)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
        if row == 0:
            combinedImage = currentRowImage 
        else:
            combinedImage = np.concatenate((combinedImage, currentRowImage), axis=0)

    #cv2.imshow('Main', combinedImage)

    #cv2.waitKey()
    #cv2.destroyAllWindows()
    path2 = path+imageOutputFolder+"outputImage"+str(outputNumber) + ".jpg"

    #cv2.imwrite(path2, combinedImage, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite(path2,combinedImage)
#oriimg = cv2.imread(filename, cv2.IMREAD_COLOR) 

with open(outputPath + outputFaceDataTextFile, 'w') as f:
  f.truncate(0) # need '0' when using r+
  for faceData in outputFaceData:
    f.write(faceData)
    f.write("\n")
'''
with open(outputPath + outputLocationDataTextFile, 'w') as f:
  f.truncate(0) # need '0' when using r+
  f.write('%d' % number)
'''
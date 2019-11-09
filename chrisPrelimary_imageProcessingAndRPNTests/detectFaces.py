#####THE FACE RECOGNITION###################################
#source: http://gregblogs.com/computer-vision-cropping-faces-from-images-using-opencv2/
# Importing the libraries
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

#saves sections of the image with faces
def facechop(image, outputFolder):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = np.copy(image)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face, (70,70))
        face_file_name = outputFolder + "face_" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, sub_face) #save each face image

    #cv2.imshow(image, img)

    return


####THE DECISION TREE###################
# Importing the dataset


def getData(datasetPath):
    # define training and test data directories


    # classes are folders in each directory with these names
    classes = ['frown', 'neutral', 'smile']
    
    # resize all images to 224 x 224
    data_transform = transforms.Compose([transforms.RandomResizedCrop(70), 
                                        transforms.ToTensor()])

    data = datasets.ImageFolder(datasetPath, transform=data_transform)
    # print out some data stats
    print('Num images: ', len(data))

    # prepare data loaders
    batch_size = len(data)
    num_workers = 0
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
    # Visualize some sample data

    # obtain one batch of training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    imagesnp = images.numpy() # convert images to numpy for display
    labelsnp = labels.numpy()
    
    #for i in range(len(imagesnp)):
    #    imagesnp[i] = np.transpose(imagesnp[i], (1, 2, 0))
    cv2.imshow('Main',np.transpose(imagesnp[3], (1, 2, 0)))
    cv2.waitKey()
    np.transpose(imagesnp, (0, 2, 3, 1))
    print(classes[labelsnp[3]])
    imagesnpList = []
    for i in imagesnp:
        imagesnpList += [np.ndarray.flatten(i)]
    imagesnpList = np.asarray(imagesnpList)

    Images_train, Images_test, Labels_train, Labels_test = train_test_split(imagesnpList, labelsnp, test_size = 0.25, random_state = 0)
    return Images_train, Images_test, Labels_train, Labels_test
    
    '''
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(len(labels)):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(classes[labels[idx]])
    '''

'''
""" Decision Trees """
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
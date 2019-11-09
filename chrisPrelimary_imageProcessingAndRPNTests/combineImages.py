import cv2
import numpy as np
import os

path =os.getcwd()
imageFolder = "\\imagesIn\\"
imageOutputFolder = "\\outputImages\\"

imageName = "1_I_1.jpg"


numRows = 3
numCols = 3

#############

imagePath = path+imageFolder+ imageName
outputPath = path + imageOutputFolder


print (imagePath)
image = cv2.imread(imagePath, cv2.IMREAD_COLOR) 



#cv2.imshow("",image)

#cv2.waitKey()

# I just resized the image to a quarter of its original size
#image = cv2.resize(image, (0, 0), None, .25, .25)



images = []
for row in range(numRows):    
    imageRow = []
    for col in range(numCols):
        imageRow += [image]
    images+=[imageRow]


combinedImage = None
for row in range(numRows):
    currentRowImage = images[row][0]

    for col in range(1,numCols):
        currentRowImage = np.concatenate((currentRowImage, images[row][col]), axis=1)
        
    if row == 0:
        combinedImage = currentRowImage 
    else:
        combinedImage = np.concatenate((combinedImage, currentRowImage), axis=0)

cv2.imshow('Main', combinedImage)

cv2.waitKey()
path2 = path+imageOutputFolder+"image2.jpg"

cv2.imwrite(path2,combinedImage)
#oriimg = cv2.imread(filename, cv2.IMREAD_COLOR) 

'''
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_vertical = np.vstack((image, image, image, image))
numpy_horizontal = np.hstack((numpy_vertical,numpy_vertical))

numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Main', image)
cv2.imshow('Numpy Vertical', numpy_vertical)
cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

cv2.waitKey()
'''
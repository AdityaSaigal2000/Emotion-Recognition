import cv2
import numpy as np
import os



path =os.getcwd()
imageFolder = "imagesIn\\"
imageName = "1_I_1.jpg"

imagePath = path+imageFolder+ imageName

print (imagePath)
image = cv2.imread(imagePath, cv2.IMREAD_COLOR) 

'''
cv2.imshow("",oriimg)

cv2.waitKey()
'''

'''
height, width, depth = oriimg.shape
imgScale = W/width
newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale

newimg = cv2.resize(oriimg,(224,224))
#cv2.imshow("Show by CV2",newimg)
#cv2.waitKey(0)
cv2.imwrite(name,newimg)


image = cv2.imread('imagesIn\\2_I_3.png')
cv2.imshow(image)
'''
# I just resized the image to a quarter of its original size
image = cv2.resize(image, (0, 0), None, .25, .25)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_vertical = np.vstack((image, grey_3_channel))
numpy_horizontal = np.hstack((image, grey_3_channel))

numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Main', image)
cv2.imshow('Numpy Vertical', numpy_vertical)
cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

cv2.waitKey()

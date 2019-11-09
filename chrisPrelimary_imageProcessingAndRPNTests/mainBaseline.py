import cv2
import numpy as np
import os
import detectFaces
from sklearn.tree import DecisionTreeClassifier


path =os.getcwd()
imageFolder = "\\faceImages\\"
imageOutputFolder = "\\faceCuts\\"
imageName = "faceReference.jpg"
datasetFolder = "\\EmotionPics\\"

#############
imagePath = path+imageFolder+ imageName
outputPath = path + imageOutputFolder
datasetPath = path + datasetFolder
image = cv2.imread(imagePath, cv2.IMREAD_COLOR) 
###################
Images_train, Images_test, Labels_train, Labels_test=detectFaces.getData(datasetPath)



'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Images_train = sc.fit_transform(Images_train)
Images_test = sc.transform(Images_test)
'''
# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(Images_train, Labels_train)

# Predicting the Test set results
y_pred = classifier.predict(Images_test)
print(y_pred)
print(Labels_test)
'''
cv2.imshow('Main', image)
cv2.waitKey()
cv2.destroyAllWindows()

detectFaces.facechop(image, outputPath)


'''


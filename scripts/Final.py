#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Importing all required labels 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import InputLayer

# Importing modules as vaiables
import tensorflow as tf
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[ ]:


import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import keras.preprocessing.image as img
from keras.applications.resnet50 import ResNet50


# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau


# In[ ]:

# Importing Dataset from drive using googlecolab
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:
# Entering into the folder which contains datasets

for dirname, _, filenames in os.walk('/content/drive/My Drive/pothole dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        #print(dirname)


# In[ ]:

# Showing all files in the train floder
os.listdir('/content/drive/My Drive/pothole dataset/train')


# In[ ]:
# Training the dataset

def make_train_data(label,DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    
        X.append(np.array(img))
        Z.append(str(label))


# In[ ]:
# Storing trained data in x & z variable
X=[]
Z=[]
IMG_SIZE=600
Plain='/content/drive/My Drive/pothole dataset/train/plain'
Pothole='/content/drive/My Drive/pothole dataset/train/pothole'

make_train_data('plain',Plain)
make_train_data('pothole',Pothole)


# In[ ]:


fig,ax=plt.subplots(2,5)
plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)
fig.set_size_inches(10,10)

for i in range(2):
    for j in range (5):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l][:])
        ax[i,j].set_title(Z[l])
        ax[i,j].set_aspect('equal')


# In[ ]:


le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,2)
print(Y)
X=np.array(X)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1337)

np.random.seed(42)
rn.seed(42)


# In[ ]:


base_model=ResNet50(include_top=False, weights='imagenet',input_shape=(256,256,3), pooling='max')
base_model.summary()


# In[ ]:


model=Sequential()
model.add(base_model)
model.add(Dropout(0.20))
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(2,activation='softmax'))


# In[ ]:


epochs=50
batch_size=128
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)
base_model.trainable=True # setting the VGG model to be trainable.
model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))


# In[ ]:


path = "/content/drive/My Drive/pothole dataset/test/pothole"
files = os.listdir(path)
files


# In[ ]:

# Testing the model

for i in tqdm(files):
    pth = os.path.join(path,i)
    X = cv2.imread(pth,cv2.IMREAD_COLOR)
    X = cv2.resize(X,(256,256))
    plt.figure()
    plt.imshow(X[:,:,::-1]) 
    plt.show()  

    X = np.array(X)
    X = np.expand_dims(X, axis=0)

    y_pred = np.round(model.predict(X))
    if y_pred[0][0] == 1:
        print("plain Road")
    else:
        print("pothole Road")


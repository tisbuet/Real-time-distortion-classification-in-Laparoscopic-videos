#!/usr/bin/env python
# coding: utf-8

# In[14]:
print("""
TEAM: BUET_ENDGAME                                  |
TEAM MEMBER 1: MD. TARIQUL ISLAM                    |   REAL-TIME DISTORTION CLASSIFICATION
TEAM MEMBER 2: SHEIKH ASIF IMRAN                    |       IN LAPAROSCOPIC VIDEOS
Email: tisbuet@gmail.com, shouborno@ieee.org        |
""")


import os
import argparse
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import cv2
from scipy.signal import convolve2d
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from keras_self_attention import SeqSelfAttention
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense,Dropout, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
np.seterr(divide='ignore', invalid='ignore')

from utils import noise_feature


# In[16]:


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("-extract", help="For training frames are extracted in './Extracted_train_data_images' directory", action="store_true")
args = parser.parse_args()

# In[13]:


if args.extract:
    print('train dataset contain videos from :')
    print(os.listdir('./train data'))
    dataset_location = './train data'
    folders = os.listdir(dataset_location)
    extracted_location = './Extracted_train_data_images'
    if not os.path.exists(extracted_location):
        os.mkdir(extracted_location)
    for file in folders:
        if not os.path.exists(os.path.join(extracted_location, file)):
            os.mkdir(os.path.join(extracted_location, file))

    for folder in folders:
        videos = os.listdir(os.path.join(dataset_location, folder))
        print('-----EXTRACTING FRAMES of {}------'.format(folder))
        for i in range(len(videos)):
            videoDir = os.path.join(dataset_location, folder, videos[i])
            outDir = os.path.join(extracted_location, folder, os.path.splitext(videos[i])[0])
            if os.path.exists(outDir):
                print('frames are already extracted')
                continue
            else:
                os.mkdir(outDir)
                os.system('ffmpeg -i "{}" -vf "select=not(mod(n\,20))" -vsync vfr {}/%01d.png'.format(videoDir, outDir));


# In[2]:


#inputFolder = sorted(os.listdir('.\Extracted_data_images'))
inputFolder = ['awgn', 'defocus_blur', 'motion_blur', 'smoke', 'uneven_illum', 'defocus_uneven', 'noise_smoke',               
                'noise_smoke_uneven', 'noise_uneven', 'smoke_uneven']
labelDict = dict()
for count, label in enumerate(inputFolder):
    labelDict[label] = count


print(labelDict)


sequence_length = 10        ## 10 frames from every videos have been extracted maintaining fixed 20 frames gap between
                              ##                                                consecutive frames


###################################
train_labelname = []
trainX = []
trainY = []

###################################
val_labelname = []
valX = []
valY = []

for folder in inputFolder:
    videoFolders = natsorted(os.listdir(os.path.join('./Extracted_train_data_images',folder)))
    print('\n---------------- Extracting Features from ' + folder + ' images ----------------\n')
    temp = 0
    for videos in tqdm(videoFolders):
        images = natsorted(glob((os.path.join('./Extracted_train_data_images',folder, videos, '*.png'))))
        
        
        xListSingleFrame = []
        count = 0
        for i in range(len(images)):
            d = cv2.imread(images[i])
            
            lab = cv2.cvtColor(d, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=5.0)
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            d = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            d = cv2.resize(d, (128,128))
            
            xListSingleFrame.append(noise_feature(d))
            
            del(d)
            if (count+1) == sequence_length:
                
                if temp <= int(len(videoFolders)*0.70):             ## 70% dataset is taken for training       
                    train_labelname.append(folder + '_' + videos)
                    trainY.append(labelDict[folder])
                    trainX.append(xListSingleFrame)
                    
                else:                                               ## 30% for validation
                    val_labelname.append(folder + '_' + videos)
                    valY.append(labelDict[folder])
                    valX.append(xListSingleFrame)
                    
                break    
            count += 1
        temp += 1
print(len(trainX), len(valX))

print('_________________  DONE  _________________')


# In[7]:


x_train = np.asarray(trainX)
x_val = np.asarray(valX)
y_train = np_utils.to_categorical(trainY).astype('float64')
y_val = np_utils.to_categorical(valY).astype('float64')

sca = []
for i in tqdm(range(x_train.shape[-1])):
    maxV = max(np.max(x_train[:,:,i]), np.max(x_val[:,:,i]))
    minV = min(np.min(x_train[:,:,i]), np.min(x_val[:,:,i]))
    sca.append([minV, maxV])
    x_train[:,:,i] = (x_train[:,:,i] - minV)/(maxV-minV)                        ## normalizing from 0 to 1
    x_val[:,:,i] = (x_val[:,:,i] - minV)/(maxV-minV)
np.save('scale_data.npy', sca)                                                  ## further saving for fitting test dataset


def myModel(x):
    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size = 3,input_shape = (x_train.shape[1], x_train.shape[2]),padding = 'same',activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(0.2))
    model.add(SeqSelfAttention(attention_width=64,attention_activation='relu'))
    model.add(LSTM(64, recurrent_dropout = 0.5,return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(x, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = None
model = myModel(y_train.shape[-1])
model.summary()

checkpoint = ModelCheckpoint('./model/trained_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]


history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=200, batch_size=128, shuffle = True, callbacks = callbacks_list)

plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right', fontsize=20)
plt.savefig('accuracy curve.png')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right', fontsize=20)
plt.savefig('loss curve.png')
plt.show()

plt.close()
#!/usr/bin/env python
# coding: utf-8

print("""
########   TEST Code   ########
""")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import pandas as pd
from glob import glob
import shutil
from tqdm import tqdm
import cv2
from natsort import natsorted

from keras_self_attention import SeqSelfAttention
from keras.models import load_model
from keras.utils import np_utils
np.seterr(divide='ignore', invalid='ignore')
from utils import noise_feature


# In[9]:


model = load_model("./model/trained_model.h5", custom_objects={'SeqSelfAttention': SeqSelfAttention})
print(model.summary())


# In[56]:


inputFolder = ['awgn', 'defocus_blur', 'motion_blur', 'smoke', 'uneven_illum', 'defocus_uneven', 'noise_smoke',               
                'noise_smoke_uneven', 'noise_uneven', 'smoke_uneven']
labelDict = dict()
for count, label in enumerate(inputFolder):
    labelDict[label] = count

print(os.listdir('./test data'))
dataset_location = './test data'
folders = natsorted(os.listdir(dataset_location))

extracted_location = './Extracted_test_data_images'

if os.path.exists(extracted_location):
    shutil.rmtree(extracted_location)
    os.mkdir(extracted_location)
else:
    os.mkdir(extracted_location)


sequence_length = 10
test_labelname = []
y_pred0 = []


sca = np.load('./scale_data.npy',allow_pickle='TRUE')


for folder in folders:
    videos = natsorted(os.listdir(os.path.join(dataset_location, folder)))
    print('\n-----EXTRACTING FRAMES of {}------\n'.format(folder))
    for ii in range(len(videos)):
        videoDir = os.path.join(dataset_location, folder, videos[ii])
        outDir = os.path.join(extracted_location, os.path.splitext(videos[ii])[0])
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        os.system('ffmpeg -i "{}" -vf "select=not(mod(n\,20))" -vsync vfr {}/%01d.png'.format(videoDir, outDir))
        
        images = natsorted(glob(os.path.join(outDir, '*.png')))
        xListSingleFrame = []
        count = 0
        for j in range(len(images)):
            d = cv2.imread(images[j])
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
                x_test= np.asarray(xListSingleFrame)[None]
                for k in range(x_test.shape[-1]):
                    minV, maxV = sca[k]
                    xx = x_test[:,:,k]
                    xx[xx>maxV] = maxV
                    xx[xx<minV] = minV
                    x_test[:,:,k] = xx
                    x_test[:,:,k] = (x_test[:,:,k] - minV)/(maxV-minV)
                y_pred0.append(model.predict(x_test).argmax(axis = 1)[0]+1)
                test_labelname.append(videos[ii])
                break
            count += 1
shutil.rmtree(extracted_location)


# In[66]:
y_pred = []
for num in y_pred0:
    if num >= 6:
        if num == 6:
            num = '2,5'
        if num == 7:
            num = '1,4'
        if num == 8:
            num = '1,4,5'
        if num == 9:
            num = '1,5'
        if num == 10:
            num = '4,5'
    else:
        num = str(num)
    y_pred.append(num)
    
    
NEWLINE_SIZE_IN_BYTES = -1 
with open('predict.txt', 'wb') as fout:
    np.savetxt(fout, y_pred,  fmt='%s')
    fout.seek(NEWLINE_SIZE_IN_BYTES, 2)
    fout.truncate()

print("""
#################

Predicted result on test dataset has been stored in 'predict.txt' in current directory

#################
""")
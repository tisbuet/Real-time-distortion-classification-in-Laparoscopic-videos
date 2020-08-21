#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from scipy.signal import convolve2d


def hbf(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                   [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    return image


def predictSAN(s_histogram, threshold):
    s_bins = len(s_histogram)
    thresh_bin = int((s_bins-1) * threshold)
    part1 = s_histogram[:thresh_bin]
    part2 = s_histogram[(thresh_bin+1):]
    sumOverall = np.sum(s_histogram)
    # print "Overall sum: ", sumOverall
    if (sumOverall <= 0):
        return [0,0]
    smoke = np.sum(part1) / float(sumOverall)
    # prediction[NO_SMOKE, SMOKE]
    prediction = [1 - smoke, smoke]
    return prediction



def getSatHisto(image, hist_height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv) # sat channel is at position 1
    histSize = image.shape[0] # s_bins
    s_ranges = [0,histSize]
    uniform = True
    accumulate = False
    s_hist = cv2.calcHist([hsv_planes[1]], [0], None, [histSize], s_ranges)
    cv2.normalize(s_hist, s_hist, 0, hist_height, cv2.NORM_MINMAX)
    return s_hist


# In[4]:


#awgn estimation

def awgn_estimation(img):
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    img = hbf(img)
    Sigma=np.sum(np.sum(np.abs(convolve2d(img, m))))
    awgn_value = Sigma*np.sqrt(0.5*np.pi)/(6*(img.shape[0]-2)*(img.shape[1]-2))
    return awgn_value


#smoke_detection
def smoke_detect(img):
    sat_histogram = getSatHisto(img, img.shape[0])
    smoke = predictSAN(sat_histogram, float(0.7))[0]
    return smoke


#Uneven_illumination_detection
def uneven_illumination(img):
    
    brightYCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    Luminance = brightYCB[:,:,0]

    num_pixels = Luminance.shape[0] * Luminance.shape[1]
    sum_pixels = np.sum(Luminance)
    LMR = sum_pixels/(num_pixels * (np.max(Luminance) - np.min(Luminance)))
    
    return LMR


#Blur detection

def defocus_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))

def myfilt(img, ker):
    new_img = img.copy()
    for i in range(img.shape[0]):
        new_img[i,:] = np.correlate(img[i,:], ker, "same")
    return new_img
def blur_estimate(img):
    img = np.double(hbf(img))
    y, x = img.shape
    Hv = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])/9
    #Hh = Hv.T
    #B_Ver = cv2.filter2D(I,-1, Hv) # blur the input image in vertical direction
    #B_Hor = cv2.filter2D(I,-1, Hh) #blur the input image in horizontal direction
    B_Ver = myfilt(img, Hv)
    B_Hor = myfilt(img.T, Hv).T

    D_F_Ver = np.abs(img[:,0:x-1] - img[:,1:x]) # variation of the input image (vertical direction)
    D_F_Hor = np.abs(img[0:y-1,:] - img[1:y,:])  # variation of the input image (horizontal direction)

    D_B_Ver = np.abs(B_Ver[:,0:x-1] - B_Ver[:,1:x]) #variation of the blured image (vertical direction)
    D_B_Hor = np.abs(B_Hor[0:y-1,:]-B_Hor[1:y,:]) #variation of the blured image (horizontal direction)
    T_Ver = D_F_Ver - D_B_Ver #difference between two vertical variations of 2 image (input and blured)
    T_Hor = D_F_Hor - D_B_Hor #difference between two horizontal variations of 2 image (input and blured)
    V_Ver = T_Ver.copy()
    V_Hor = T_Hor.copy()
    V_Ver[V_Ver < 0] = 0
    V_Hor[V_Hor < 0] = 0


    S_D_Ver = np.sum(D_F_Ver[1:y-1,1:x-1])
    S_D_Hor = np.sum(D_F_Hor[1:y-1,1:x-1])
    S_V_Ver = np.sum(V_Ver[1:y-1,1:x-1])
    S_V_Hor = np.sum(V_Hor[1:y-1,1:x-1])
    blur_F_Ver = (S_D_Ver-S_V_Ver)/S_D_Ver
    blur_F_Hor = (S_D_Hor-S_V_Hor)/S_D_Hor
    a = max(blur_F_Ver,blur_F_Hor)
    return a
    
def noise_feature(img):
    return [awgn_estimation(img), defocus_blur(img), blur_estimate(img), smoke_detect(img), uneven_illumination(img)]
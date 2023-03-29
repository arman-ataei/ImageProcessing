#preprocessing
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np # linear algebra
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

# a funtion for padding segmented images 


borderType = cv2.BORDER_CONSTANT
def pad(src): 
    top = int(0.05 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.15 * src.shape[1])  # shape[1] = cols
    right = left
    des=cv2.copyMakeBorder(src, top, bottom, left+1, right, borderType, None,255)
    return cv2.bitwise_not(des)
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

#segmentation and removing horizontal lines from the images


def Segmentation(img,imgStr, x, y):
    for i in range(5): #(hard coded) segmentation and creating labled data 
        x.append(pad(img[:,(30+23*i):(30+23*(i+1))]))
        y.append(imgStr[-9+i])

def RemoveLineAndSegment(imglst, x, y, z):
    kernel =np.ones((3,1),np.uint8)
    for image in imglst:
        im=cv2.imread(str(image),0) #removing lines from the images
        threshold=cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 5) 
        erosion =cv2.dilate(threshold,kernel,iterations=2)
        z.append(erosion)
        s=str(image)
        Segmentation(erosion, s, x, y)

def Segment(img,imgStr):
    x=[]
    y=[]
    for i in range(5):
        x.append(pad(img[:,(30+23*i):(30+23*(i+1))]))
        y.append(imgStr[-9+i])
    return(x,y)
    

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#End of preprocess
#ref: https://www.kaggle.com/code/xiaowangiiiii/ocr-with-opencv-92-9-accuracy-on-captcha-34d0d9/notebook
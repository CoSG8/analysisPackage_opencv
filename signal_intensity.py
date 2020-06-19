#!/usr/bin/env python
# coding: utf-8
# signal_intensity.py
# ver. 1.0	2020.06.05

# In[Import]:
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import numpy as np
import os
import tkinter
import tkinter.filedialog
from matplotlib import pyplot as plt


# In[Open files]:

root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname('__file__'))
files = tkinter.filedialog.askopenfilenames(initialdir = iDir)
videoPath = list(files)

fName = os.path.basename(os.path.splitext(videoPath[0])[0])

img = cv2.imread(videoPath[0])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite(fName + "_original.png",img)


# In[Masking]:
# HSV色空間に変換
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 赤色のHSVの値域1
hsv_min = np.array([0,64,0])
hsv_max = np.array([30,255,255])
mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

# 赤色のHSVの値域2
hsv_min = np.array([150,64,0])
hsv_max = np.array([179,255,255])
mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

# 赤色領域のマスク（255：赤色、0：赤色以外）    
mask = mask1 + mask2

# マスキング処理
img_mask = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB))
cv2.imwrite(fName + "_mask.png",img_mask)


# In[Gray scale]:

gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
cv2.imwrite(fName +"_gray.png",gray)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.plot(hist)
plt.xlim([0,50])
plt.ylim([0,700000])


# In[Calculate intensity]

thresh = gray.mean()+gray.std()
img_size = gray.size

gray_th = gray.copy()
gray_th[gray_th < thresh] = 0
plt.imshow(gray_th,cmap='gray')
cv2.imwrite(fName +"_gray_th.png",gray_th)
hist_th = cv2.calcHist([gray_th],[0],None,[256],[0,256])
plt.plot(hist_th)
plt.xlim([0,50])
plt.ylim([0,700000])

signalInt = len(gray_th[gray_th>0])/img_size *100

print(signalInt)


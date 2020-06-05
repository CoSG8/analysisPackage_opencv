#!/usr/bin/env python
# coding: utf-8
# watershed_test_v5.ipynb
# ver. 5.0	2020.06.05

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
img = cv2.imread(videoPath[0])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite("./original.png",img)


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
cv2.imwrite("./mask.png",img_mask)


# In[Gray scale]:

gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
cv2.imwrite("./gray.png",gray)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)
hist_clahe  = cv2.calcHist([gray_clahe ],[0],None,[256],[0,256])
plt.plot(hist_clahe)
cv2.imwrite("./hist.png",hist_clahe)

plt.imshow(gray_clahe,cmap='gray')
cv2.imwrite("./gray_clahe.png",gray_clahe)


# In[Binarization]:
 
thresh, bin_img = cv2.threshold(gray_clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# thresh, bin_img = cv2.threshold(gray_clahe,110,255,cv2.THRESH_BINARY)
plt.imshow(bin_img,cmap='gray')
print(thresh)
cv2.imwrite("./bin_img.png",bin_img)


# In[Mathematical Morphology (noise removal)]:

kernel = np.ones((4,4),np.uint8)
# kernel = np.ones((3,3),np.uint8)
# kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations = 2)
plt.imshow(opening, cmap='gray')
cv2.imwrite("./opening.png",opening)


# In[Finding sure background area]:

sure_bg = cv2.dilate(opening,kernel,iterations=2)
plt.imshow(sure_bg, cmap='gray')
cv2.imwrite("./sure_bg.png",sure_bg)


# In[Finding sure foreground area]:

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform)
cv2.imwrite("./dist_transform.png",dist_transform)


# In[Threshold]:

ret,sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
print("Threshold: ", ret)
plt.imshow(sure_fg, cmap='gray')
cv2.imwrite("./sure_fg.png",sure_fg)


# In[Finding unknown region]:

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
plt.imshow(unknown, cmap='gray')
cv2.imwrite("./unknown.png",unknown)


# In[Marker labelling]:

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
plt.imshow(markers, cmap='jet')
cv2.imwrite("./markers.png",markers)


# In[Watershed]:

markers = cv2.watershed(img, markers)
plt.imshow(markers, cmap='jet')
cv2.imwrite("./watershed.png",markers)


# In[find contor]:

contour = img.copy()
contour[markers == -1] = [255,255,255]
plt.imshow(cv2.cvtColor(contour, cv2.COLOR_BGR2RGB))
cv2.imwrite("./contour.png",contour)


contour_g = gray_clahe.copy()
contour_g = cv2.cvtColor(contour_g, cv2.COLOR_GRAY2BGR)
contour_g[markers == -1] = [0,0,255]
plt.imshow(cv2.cvtColor(contour_g, cv2.COLOR_BGR2RGB))
cv2.imwrite("./contour_g.png",contour_g)


np.unique(markers, return_counts=True)


# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:54:24 2021

@author: Weibo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:39:40 2019
    load map and visualize 

@author: Weibo

"""
# %% Import necessary packages
import os 
import sys
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# path = 'I:/Hongyu/'
path = '/media/weibo/Work/Wang/'
cam = cv2.VideoCapture(path + 'w-1.avi')

try:      
    # creating a folder named data
    if not os.path.exists(path + 'frame_orig'):
        os.makedirs(path + 'frame_orig')
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
pathFrame = path + 'frame_orig/'
cframe = 0
while(cframe < 4000):
    ret, frame = cam.read()
    if ret:
        name = pathFrame + 'frame' + '{0:04d}'.format(cframe) + '.jpg'
        cv2.imwrite(name, frame)
        cframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()

# %% convert image to data
# import an image from a file
img = Image.open(pathFrame + 'frame0000.jpg')
# extracting pixel map
imgarr = img.load()
# width and height of the image
width0, height0 = img.size
grayarr = np.ones((width0, height0)) * 255
for i in range(width0):
    for j in range(height0):
        # get the RGB pixel value
        rc, gc, bc = img.getpixel((i, j))
        # convert color to grayscale using formula
        grayscale = int(0.299 * rc + 0.587 * gc + 0.114 * bc)
        grayarr[i, height0 - 1 - j] = grayscale 
# imgdata = np.asarray(imgarr, dtype='int32')
grayarr = np.transpose(grayarr)
grayfil = grayarr[0, 0]
print("the cutoff value is ", grayfil)
ind = np.argwhere(grayarr == grayfil)
# graynew = grayarr[58:152 , :]
newrg = 513
graynew = grayarr[20:240, 401:800] / 255 * newrg
height = np.shape(graynew)[0]
width = np.shape(graynew)[1]
# %% plot contour to examine the conversion
wid = width # width  # 100
hei = height  #  40
x1 = np.linspace(0, wid - 1, width)
y1 = np.linspace(0, hei - 1, height)
x2, y2 = np.meshgrid(x1, y1)
graynew[:50, 350:] = 0.0
textsize = 12
numsize = 10
fig, ax = plt.subplots(figsize=(6.4, 2.8))
matplotlib.rc("font", size=textsize)
rg1 = np.linspace(0, newrg, 64)
cbar = ax.contourf(x2, y2, graynew, cmap="rainbow", levels=rg1, extend='both')  # rainbow_r
# ax.set_xlim(-10.0, 30.0)
# ax.set_ylim(-3.0, 10.0)
ax.tick_params(labelsize=numsize)
ax.set_xlabel(r"$x$", fontsize=textsize)
ax.set_ylabel(r"$y$", fontsize=textsize)
# plt.gca().set_aspect("equal", adjustable="box")
# Add colorbar
rg2 = np.linspace(0, newrg, 4)
cbaxes = fig.add_axes([0.17, 0.68, 0.20, 0.07])  # x, y, width, height
cbaxes.tick_params(labelsize=numsize)
cbar = plt.colorbar(cbar, cax=cbaxes, extendrect='False',
                    orientation="horizontal", ticks=rg2)
cbar.set_label(
    r"$\langle \rho \rangle/\rho_{\infty}$", rotation=0, fontsize=textsize
)
plt.savefig(path + "test.png", dpi=600, bbox_inches="tight")
plt.show()
# %% save dataframe
dirs = sorted(os.listdir(pathFrame))
for kk in range(np.size(dirs)):
    img = Image.open(pathFrame + dirs[kk])
    imgarr = img.load()
    width, height = img.size
    grayarr = np.ones((width, height)) * 255
    if (width == width0) and (height == height0):
        pass
    else:
        sys.exit("The shape of " + dirs[i] + "is not compatible!")
    for i in range(width):
        for j in range(height):
            rc, gc, bc = img.getpixel((i, j))
            grayscale = int(0.299 * rc + 0.587 * gc + 0.114 * bc)
            grayarr[i, height - 1 - j] = grayscale 
    grayarr = np.transpose(grayarr)
    graynew = grayarr[20:240, 401:800] / 255 * newrg
    graynew[:50, 350:] = 0.0
    frame_val = np.vstack([x2.ravel(order='F'), y2.ravel(order='F')])
    frame_val = np.vstack([frame_val, graynew.ravel(order='F')])
    datafra = pd.DataFrame(frame_val.T, columns=['x', 'y', 'gray'])
    datafra.to_hdf(path + 'framedata/' + '{0:04d}'.format(kk) + '.h5', 
                   'w', format='fixed')

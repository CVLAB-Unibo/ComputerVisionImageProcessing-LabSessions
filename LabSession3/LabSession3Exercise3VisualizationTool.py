#!/usr/bin/env python
# coding: utf-8

# # Visualization Tool
# ## Computer Vision and Image Processing - Lab Session 3 Exercises
# ### Prof: Luigi Di Stefano, luigi.distefano@unibo.it
# ### Tutor: Pierluigi Zama Ramirez, pierluigi.zama@unibo.it

# Use this tool to get the **coordinate** of pixels. Modify the variable *image_path* with the absolute path to an image. Run the code below and you will se a pop-up windows opening. Click on a pixel on the image and you will get the coordinate **(row,column)** of that pixel.

# Define here your path
image_path = "es2/pen.jpg"


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

fig = plt.figure(figsize=(20,30))

img=cv2.imread(image_path)

def onclick(event):
    ix, iy = event.xdata, event.ydata
    print("Coordinate clicked pixel (row,column): [{},{}]".format(int(round(ix)), int(round(iy))))

cid = fig.canvas.mpl_connect('button_press_event', onclick)

imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()





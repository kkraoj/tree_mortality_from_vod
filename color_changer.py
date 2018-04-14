# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:29:46 2018

@author: kkrao
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
from PIL import Image 

name='jet'
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

sns.set_style('white')
zoom = 2
fig, ax =plt.subplots(figsize=(4*zoom,0.25*zoom))
ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
plt.axis('off')
plt.savefig('color_changer_example.png',bbox_inches='tight')

#===================================================================
img=mpimg.imread('color_changer_example.png')
                                
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    
gray = rgb2gray(img)  
fig, ax =plt.subplots(figsize=(4*zoom,0.25*zoom))  
imgplot = plt.imshow(gray,cmap='Greys')
plt.axis('off')
#fig.subplots_adjust() 
#ax.tick_params(which='both',)


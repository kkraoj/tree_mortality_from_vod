# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:47:23 2018

@author: kkrao
"""
import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors
from scipy.spatial import cKDTree
from ipywidgets import interact
import matplotlib.image as mpimg

userhome = os.path.expanduser('~')
desktop = userhome + '/Desktop/'

os.chdir(desktop)
img = mpimg.imread('jet_sample.png')[:,:,:3]

#@interact(sub=(0, 500), d=(0,1,0.05))
def convert(sub=256,d=0.2, cin='jet', cout='viridis'):
    viridis = plt.get_cmap(cout)
    jet = plt.get_cmap(cin)
    jet256 = colors.makeMappingArray(sub, jet)[:, :3]
    K = cKDTree(jet256)
    oshape = img.shape
    img_data = img.reshape((-1,3))
    res = K.query(img_data, distance_upper_bound=d)
    indices = res[1]
    l = len(jet256)
    indices = indices.reshape(oshape[:2])
    remapped = indices

    indices.max()

    mask = (indices == l)

    remapped = remapped / (l-1)
    mask = np.stack( [mask]*3, axis=-1)

    blend = np.where(mask, img, viridis(remapped)[:,:,:3])
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.imshow(blend)
    
convert()

#!open viridize.png

#img = mpimg.imread('/Users/bussonniermatthias/Desktop/download-2.png')[:,:,:3]
plt.imshow(img)


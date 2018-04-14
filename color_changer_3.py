# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:11:47 2018

@author: kkrao
"""

from __future__ import division
import os
import Image

palette = []
levels = 8
stepsize = 256 // levels
for i in range(256):
    v = i // stepsize * stepsize
    palette.extend((v, v, v))

assert len(palette) == 768

os.chdir('Desktop')          
original_path = 'jet_sample.png'
original = Image.open(original_path)
converted = Image.new('P', original.size)
converted.putpalette(palette)
converted.paste(original, (0, 0))
converted.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for generating videos

@author: weibo
"""


# %% Load necessary module
import imageio
from natsort import natsorted  # , natsort
import os

# %% convert figures to animation
path_in = "/media/weibo/VID1/BFS_M1.7L/K-H/0/"
path_out = "/media/weibo/VID1/BFS_M1.7L/K-H/"
outfile = "vortex.mp4"
dirs = os.listdir(path_in)
dirs = natsorted(dirs, key=lambda y: y.lower())
with imageio.get_writer(path_out + outfile, mode='I', fps=1) as writer:
    for filename in dirs:
        image = imageio.imread(path_in + filename)
        writer.append_data(image)

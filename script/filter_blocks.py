# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    Extract a uniform volume for 3D analysis

@author: weibo
"""
# %% Load necessary module
import plt2pandas as p2p
import pandas as pd
import numpy as np
import tecplot as tp
from glob import glob

# %% filter the flow domain
path = "/media/weibo/IM2/FFS_M1.7TB/"
pathin = path + "TP_stat/"
pathout = path

cube = [(-10.0, 30.0), (0.0, 6.0), (-8, 8)]
FileId = p2p.extract_zone(pathin, cube)
FileId['name'].to_csv(pathout+'names.dat', index=False)
FileId.to_csv(pathout+'VortexList1.dat', index=False, sep=' ')
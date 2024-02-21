#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 21 14:48:11 2024
    This code for coverting grid tecplot format to plot3d format

@author: weibo
"""
# %% Load necessary module
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.io import FortranFile
import os
import sys

# %%
def gridplot3d(path, filenm, vars, option='3d'):
    data = pd.read_csv(path + filenm, sep='\t')
    vars = ['x', 'y', 'z']
    grid = 'grid.grd'
    xa = np.unique(data['x'])
    ya = np.unique(data['y'])
    za = np.unique(data['z'])
    xmat, ymat, zmat = np.meshgrid(xa, ya ,za)

    # organize grid
    df_grid = np.vstack((
        xmat.flatten(order='C'),
        ymat.flatten(order='C'),
        zmat.flatten(order='C')
    ))

    dim = np.array([np.size(xa), np.size(ya), np.size(za)], dtype=np.int32)
    # output grid
    file = FortranFile(path + grid, 'w')
    file.write_record(np.array([1], dtype=np.int32))
    file.write_record(np.array([dim], dtype=np.int32))
    file.write_record(dim)
    file.write_record(df_grid)
    file.close()    

    return(df_grid)

def interp2d(x, y, z):
    xa = np.unique(x)
    ya = np.unique(y)
    xmat, ymat = np.meshgrid(xa, ya)
    zmat = griddata((x, y), z, (xmat, ymat))
    return(zmat)     

# %%
if __name__ == "__main__":
    # path = 'E:/cases/'
    # filenm = 'slice_3d.dat'
    path = 'E:/cases/flat_base/'  
    filenm = 'slice.txt'
    vars = ['x', 'y', 'rho', 'u', 'v', 'w', 'T', 'p']
    grid2plot3d(path, filenm, vars, option='3d')
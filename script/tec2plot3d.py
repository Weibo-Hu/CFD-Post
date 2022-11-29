#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:48:11 2022
    This code for coverting tecplot format to plot3d format

@author: weibo
"""
# %% Load necessary module
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.io import FortranFile

# %%
def csv2plot3d(path, filenm, vars, option='2d', input_nm=None, skip=1):
    if input_nm is None:
        data = pd.read_csv(path + filenm, sep=',', index_col=False)
    else:
        data = pd.read_csv(path + filenm, sep=',', index_col=False,
                           header=0, names=input_nm)
    grid = 'grid.grd'
    solu = 'uvw.q'
    xa = np.unique(data['x'][::skip])
    ya = np.unique(data['y'])
    xmat, ymat = np.meshgrid(xa, ya)
    # organize grid
    df_grid = np.vstack((
        xmat.flatten(order='C'), 
        ymat.flatten(order='C')
        ))   
    dim = np.array([np.size(xa), np.size(ya)], dtype=np.int32)
    if option=='3d':
        zmat = griddata((data['x'], data['y']), data['z'], (xmat, ymat))
        df_grid = np.vstack((df_grid, zmat.flatten(order='C')))
        dim = np.append(dim, 1)
    # output grid
    file = FortranFile(path + grid, 'w')
    file.write_record(np.array([1], dtype=np.int32))
    file.write_record(np.array([dim], dtype=np.int32))
    file.write_record(df_grid)
    file.close()
    # organize solution
    for j in range(np.size(vars)):
        umat = griddata((data['x'], data['y']), data[vars[j]],
                        (xmat, ymat))
        if j == 0:
            df_data = umat.flatten(order='C')
        else:
            df_data = np.vstack((df_data, umat.flatten(order='C')))
    nvar = np.size(vars)
    dim_var = np.append(dim, nvar)
    file = FortranFile(path + solu, 'w')
    file.write_record(np.array([1], dtype=np.int32))
    file.write_record(np.array([dim_var], dtype=np.int32))
    file.write_record(df_data)
    file.close()

    return(data)

def interp2d(x, y, z):
    xa = np.unique(x)
    ya = np.unique(y)
    xmat, ymat = np.meshgrid(xa, ya)
    zmat = griddata((x, y), z, (xmat, ymat))
    return(zmat)     


path = 'E:/cases/'
filenm = 'slice_3d.dat'
nms = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T']
vars = ['rho', 'u', 'v', 'w', 'T', 'p']
csv2plot3d(path, filenm, vars, option='3d', input_nm=nms, skip=2)
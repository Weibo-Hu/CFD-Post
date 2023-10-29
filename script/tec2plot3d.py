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
import os
import sys

# %%
def csv2plot3d(path, filenm, vars, option='2d', input_nm=None, skip=1):
    ext_nm = os.path.splitext(filenm)[-1]
    if input_nm is None:
        if ext_nm == '.dat':
            data = pd.read_csv(path + filenm, sep=',', index_col=False)
        if ext_nm == '.h5':
            data = pd.read_hdf(path + filenm, index_col=False)
    else:
        if ext_nm == '.dat':
            data = pd.read_csv(path + filenm, sep=',', index_col=False,
                               header=0, names=input_nm)
        if ext_nm == '.h5':
            data = pd.read_hdf(path + filenm, index_col=False,
                               header=0, names=input_nm)            
    grid = 'grid.xyz'
    solu = 'uvw3D.q'
    xa = np.unique(data['x'])[::skip]
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

def create_meanval(path, filenm):
    if not os.path.isfile(path + filenm):
        sys.exit('file is not exist!')
    df = pd.read_hdf(path + filenm)
    df['u']   = df['<u>']
    df['v']   = df['<v>']
    df['w']   = df['<w>']
    df['rho'] = df['<rho>']
    df['p']   = df['<p>']
    df['T']   = df['<T>']
    df.to_hdf(path + filenm, 'w', format='fixed')
    return(df)

# %%
if __name__ == "__main__":
    # path = 'E:/cases/'
    # filenm = 'slice_3d.dat'
    path = 'E:/cases/flat_base/'  
    create_meanval(path + 'MeanFlow/', 'MeanFlow.h5')
    filenm = 'MeanFlow/MeanFlow.h5'
    nms = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T']
    vars = ['rho', 'u', 'v', 'w', 'T', 'p']
    csv2plot3d(path, filenm, vars, option='2d', input_nm=nms, skip=4)
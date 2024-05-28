#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 21 14:48:11 2024
    This code for coverting grid tecplot format to plot3d format

@author: weibo
"""
# %% Load necessary module
import numpy as np
from scipy.interpolate import griddata
from scipy.io import FortranFile
import os
from source import pytecio as pt

# %%
# convert tecplot grid to plot3d grid
def grid2plot3d(path, outpath, option='3d'):
    vars = ['x', 'y', 'z']
    dirs = os.listdir(path)
    for i in range(np.size(dirs)):
        data, _ = pt.ReadSinglePlt(path + dirs[i], vars)
        grid = 'grid_' + "%06d" % i + '.xyz'
        print('processing ' + dirs[i])
        xa = np.unique(data['x'])
        ya = np.unique(data['y'])
        za = np.unique(data['z'])
        ymat, zmat, xmat = np.meshgrid(ya, za, xa)
        # organize grid
        df_grid = np.vstack((
            xmat.flatten(order='C'), 
            ymat.flatten(order='C'),
            zmat.flatten(order='C')
            ))   
        # df_grid = np.vstack((df_grid, zmat.flatten(order='C')))
        dim = np.array([np.size(xa), np.size(ya), np.size(za)], dtype=np.int32)
        # output grid
        file = FortranFile(outpath + grid, 'w')
        file.write_record(np.array([1], dtype=np.int32))
        file.write_record(dim)
        file.write_record(df_grid)
        file.close() 
    return (df_grid)


def interp2d(x, y, z):
    xa = np.unique(x)
    ya = np.unique(y)
    xmat, ymat = np.meshgrid(xa, ya)
    zmat = griddata((x, y), z, (xmat, ymat))
    return (zmat)     


def select_block(inpath, outpath, zone):
    dirs = os.listdir(inpath)
    namepick = []
    namedrop = []
    for i in range(np.size(dirs)):
        with open (inpath + dirs[i]) as f:
            lines = f.readlines()
        xrange = np.float64(lines[13].split()[2:5:2])
        yrange = np.float64(lines[14].split()[2:5:2])
        zrange = np.float64(lines[15].split()[2:5:2])
        cond1 = (xrange[0] >= zone[0][0]) & (xrange[0] <= zone[0][1])
        cond2 = (yrange[0] >= zone[1][0]) & (yrange[0] <= zone[1][1])
        cond3 = (zrange[0] >= zone[2][0]) & (zrange[0] <= zone[2][1])
        if (cond1 & cond2 & cond3):
            namepick.append(dirs[i])
        else:
            namedrop.append(dirs[i])
    # list of names
    with open(outpath+'namedrop.dat', 'w') as fp:
        fp.write('\n'.join(namedrop))
    with open(outpath+'namepick.dat', 'w') as fp:
        fp.write('\n'.join(namepick))
    return (namedrop)

def edit_block(inpath, outpath, zarr):
    dirs = os.listdir(inpath)
    zstr = [str(j) for j in zarr]
    for i in range(np.size(dirs)):
        with open (inpath + dirs[i]) as f:
            lines = f.readlines() 
        nz_old = lines[18].split()[2]
        if nz_old == '64':
            lines[18] = lines[18].replace(nz_old, zstr[0])
        elif nz_old == '32':
            lines[18] = lines[18].replace(nz_old, zstr[1])
        elif nz_old == '16':
            lines[18] = lines[18].replace(nz_old, zstr[2])
        elif nz_old == '8':
            lines[18] = lines[18].replace(nz_old, zstr[3])                
        else:
            print('there is an error for ' + dirs[i])
        with open(outpath+dirs[i], 'w') as fp:
            fp.write("".join(lines))


# %%
if __name__ == "__main__":
    # path = 'E:/cases/'
    # filenm = 'slice_3d.dat'

    inpath = 'D:/cases/ramp_st1/grid/'
    outpath = 'D:/cases/ramp_st1/grid_new/'
    zone = np.array(
        [[-225, 87.5],
         [0, 48],
         [-4, 4]]
    )
    namedrop = select_block(inpath, outpath, zone)
    for i in range(np.size(namedrop)):
        os.remove(inpath + namedrop[i])

    zarr0 = np.array([64, 32, 16, 8])
    zarr = np.array([80, 40, 20, 10])
    edit_block(inpath, outpath, zarr)
    path = '/media/work/data1/flap2/flapping_wall/'
    grid2plot3d(path+'TP_grid/', path+'grid/',  option='3d')
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
import sys
import re

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

def edit_block(inpath, outpath, znew, zold=None, xarr=None, opt=0):
    dirs = sorted(os.listdir(inpath))
    zstr = [str(j) for j in znew]
    if opt == 0:
        lxrg = 13
        lxno = 16
        lzno = 18
    elif opt == 1:
        lxrg = 3
        lxno = 7
        lzno = 9
    else:
        sys.exit('option error')

    for i in range(np.size(dirs)):
        print('process the block of ', dirs[i])
        with open (inpath + dirs[i]) as f:
            lines = f.readlines()
        zpt_ln = re.split('[,\s]', lines[lzno])
        zpt_ln = list(filter(None, zpt_ln))
        nz_old = zpt_ln[2]
        if zold is not None:
            if isinstance(zold[0], np.int64) is not True:
                sys.exit("the type of zno is not int!!!")
            if nz_old == str(zold[0]):
                lines[lzno] = lines[lzno].replace(nz_old, zstr[0])
            elif nz_old == str(zold[1]):
                lines[lzno] = lines[lzno].replace(nz_old, zstr[1])
            elif nz_old == str(zold[2]):
                lines[lzno] = lines[lzno].replace(nz_old, zstr[2])
            elif nz_old == str(zold[3]):
                lines[lzno] = lines[lzno].replace(nz_old, zstr[3])             
            else:
                print('there is an error for ' + dirs[i])
        if xarr is not None:
            if isinstance(xarr[0], np.float64) is not True:
                sys.exit("the type of xarr is not float!!!")
            xrg_ln = re.split('[,\s]', lines[lxrg])
            xrg_ln = list(filter(None, xrg_ln))
            xlc1 = float(xrg_ln[2])
            xlc2 = float(xrg_ln[3])
            xpt_ln = re.split('[,\s]', lines[lxno])
            xpt_ln = list(filter(None, xpt_ln))
            xpts = float(xpt_ln[2])
            xspc = (xlc2 - xlc1) / xpts
            print('the x-space of grid is ', xspc)
            for jj in range(np.size(xarr)):
                if xspc == xarr[jj]:
                    lines[lzno] = lines[lzno].replace(nz_old, zstr[jj])
            # if xspc == xarr[0]:
            #    lines[lzno] = lines[lzno].replace(nz_old, zstr[0])
            # elif xspc == xarr[1]:
            #    lines[lzno] = lines[lzno].replace(nz_old, zstr[1])
            # elif xspc == xarr[2]:
            #    lines[lzno] = lines[lzno].replace(nz_old, zstr[2])
            # elif xspc == xarr[3]:
            #    lines[lzno] = lines[lzno].replace(nz_old, zstr[3])      
            if xspc not in xarr:    
                print('there is an error for ' + dirs[i])
        with open(outpath+dirs[i], 'w') as fp:
            fp.write("".join(lines))


# %%
if __name__ == "__main__":
    # path = 'E:/cases/'
    # filenm = 'slice_3d.dat'

    inpath = '/media/weibo/VID2/ramp_st17/grid1338/'
    outpath = '/media/weibo/VID2/ramp_st17/grid1338_new/'
    zone = np.array(
        [[-225, 87.5],
         [0, 48],
         [-4, 4]]
    )
    namedrop = select_block(inpath, outpath, zone)
    for i in range(np.size(namedrop)):
        os.remove(inpath + namedrop[i])

    zod = np.array([60, 40, 20, 10])
    zarr = np.array([6, 12, 24, 48, 48, 48, 48, 48])
    xarr = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    edit_block(inpath, outpath, zarr, xarr=xarr, opt=0)
    path = '/media/work/data1/flap2/flapping_wall/'
    grid2plot3d(path+'TP_grid/', path+'grid/',  option='3d')

# %%
inpath = '/media/weibo/VID2/ramp_st17/grid/'
outpath = '/media/weibo/VID2/ramp_st17/grid_new/'
dirs = sorted(os.listdir(inpath))

lxrg = 13
lxno = 16
lzrg = 15
lzno = 18

for i in range(np.size(dirs)):
    print('process the block of ', dirs[i])
    with open (inpath + dirs[i]) as f:
        lines = f.readlines()
    
    zrg_ln = re.split('[,\s]', lines[lzrg])
    zrg_ln = list(filter(None, zrg_ln))
    if float(zrg_ln[2]) == 0.0:
        print('there is a mismatch for ', dirs[i])
    else:
        if float(zrg_ln[3]) == 0.0:
            zpt_ln = re.split('[,\s]', lines[lzno])
            zpt_ln = list(filter(None, zpt_ln))
            nz_old = zpt_ln[2]
            if int(nz_old) == 20:
                lines[lzno] = lines[lzno].replace(nz_old, '40')
                print("replace ", dirs[i])
            else:
                print("there is no change for ", dirs[i])
    with open(outpath+dirs[i], 'w') as fp:
        fp.write("".join(lines))

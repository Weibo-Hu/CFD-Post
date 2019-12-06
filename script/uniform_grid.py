#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    Interpolate data to a uniform grid

@author: weibo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import plt2pandas as p2p

path = '/media/weibo/Data2/FFS_CFX_INCA_test1/inca_out/'

df = pd.read_csv(path + 'initial_data.dat', sep=' ', index_col=False,
                 skiprows=[1,2,3], skipinitialspace=True)
slc = df.query('x>=-100.0 & x <=80.0 & y<=40.0')
xx = np.arange(-80.0, 60.0 + 0.125, 0.125)
yy = np.arange(0.0, 40.0 + 0.0625, 0.0625)
x, y = np.meshgrid(xx, yy)
var = 'rho'
rho = griddata((slc.x, slc.y), slc[var], (x, y))
print("rho_max=", np.max(slc[var]))
print("rho_min=", np.min(slc[var]))
corner = (x > 0.0) & (y < 3.0)
rho[corner] = np.nan
# rho[corner] = np.nan
# %%
fig, ax = plt.subplots(figsize=(6.4, 2.4))
rg1 = np.linspace(0.2, 1.4, 13)
rg2 = np.linspace(0.2, 1.4, 4)
cbar = ax.contourf(x, y, rho, cmap="rainbow", levels=rg1, extend='both')  # rainbow_r
plt.colorbar(cbar, orientation="horizontal", extendrect='False', ticks=rg2)
ax.set_xlim(-80.0, 60.0)
ax.set_ylim(0.0, 40.0)
ax.set_xlabel(r"$x/\delta_0$", fontsize=12)
ax.set_ylabel(r"$y/\delta_0$", fontsize=12)
plt.gca().set_aspect("equal", adjustable="box")
plt.savefig(path + "initial.svg", bbox_inches="tight")
plt.show()


# %% 
path = '/media/weibo/Data2/FFS_M1.7L2/'
df1 = p2p.ReadAllINCAResults(path, FileName=path+'initial1.plt')
grouped = df1.groupby(['x', 'y'])
df0 = grouped.mean().reset_index()
volume = [(-70, 40.0), (0.0, 6.0)]
dxyz = [0.015625, 0.0078125, 0.125]
#volume = [(-70, 40.0), (6.0, 33.0)]
#dxyz = [0.0625, 0.0625, 0.125]
xval = np.arange(volume[0][0], volume[0][1] + dxyz[0], dxyz[0])
yval = np.arange(volume[1][0], volume[1][1] + dxyz[1], dxyz[1])
x, y = np.meshgrid(xval, yval)
z = np.zeros(np.shape(x))

name = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T']
u = griddata((df0.x, df0.y), df0['u'], (x, y), fill_value=0, method='cubic')
v = griddata((df0.x, df0.y), df0['v'], (x, y), fill_value=0, method='cubic')
w = griddata((df0.x, df0.y), df0['w'], (x, y), fill_value=0, method='cubic')
rho = griddata((df0.x, df0.y), df0['rho'], (x, y),fill_value=0,method='cubic')
p = griddata((df0.x, df0.y), df0['p'], (x, y), fill_value=0, method='cubic')
T = griddata((df0.x, df0.y), df0['T'], (x, y), fill_value=0, method='cubic')
 
xx = x.reshape(-1, 1)
yy = y.reshape(-1, 1)
zz = z.reshape(-1, 1)
uu = u.reshape(-1, 1)
vv = v.reshape(-1, 1)
ww = w.reshape(-1, 1)
rhorho = rho.reshape(-1, 1)
pp = p.reshape(-1, 1)
TT = T.reshape(-1, 1)

var1 = np.column_stack((xx, yy, zz, uu, vv, ww, rhorho, pp, TT))
var2 = np.copy(var1)
var1[:, 2] = -8.0
var2[: ,2] = 8.0
var = np.row_stack((var1, var2))
df = pd.DataFrame(var, columns=name)

filename = "initial_data"
df1 = df.query("x<=0.0 & y>=0.0")
p2p.frame2tec3d(df1, path, filename + 'A', zname=1, stime=0.0)
p2p.tec2plt(path, filename + 'A')

# behind the step
df2 = df.query("x>=0.0 & y>=3.0")
p2p.frame2tec3d(df2, path, filename + 'B', zname=2, stime=0.0)
p2p.tec2plt(path, filename + 'B')

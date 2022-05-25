# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    Extract a uniform volume for 3D analysis

@author: weibo
"""
# %% Load necessary module
import plt2pandas as p2p
import time
import pandas as pd
import numpy as np
import tecplot as tp
from glob import glob

# %% Extract 3D flow domain
path = '/home/weibohu/weibo/FFS_M1.7TB1/'
pathin = path + "3Domain/"
pathout = path + 'hdf5/'
VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    'rho',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'L2-criterion',
    'T',
    '|gradp|'
]
#    'ux',
#    'uy',
#    'uz',
#    'vx',
#    'vy',
#    'vz',
#    'wx',
#    'wy',
#    'wz',
#    'rhox',
#    'rhoy',
#    'rhoz',
#    'px',
#    'py',
#    'pz'
#]

equ = [
    '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)'
]
#    '{ux} = ddx({u})',
#    '{uy} = ddy({u})',
#    '{uz} = ddz({u})',
#    '{vx} = ddx({v})',
#    '{vy} = ddy({v})',
#    '{vz} = ddz({v})',
#    '{wx} = ddx({w})',
#    '{wy} = ddy({w})',
#    '{wz} = ddz({w})',
#    '{rhox} = ddx({rho})',
#    '{rhoy} = ddy({rho})',
#    '{rhoz} = ddz({rho})',
#    '{px} = ddx({p})',
#    '{py} = ddy({p})',
#    '{pz} = ddz({p})',
#]

start = time.time()

volume = [(-30.0, 15.0), (0.0, 12.0), (-8, 8)]
dxyz = [0.25, 0.125, 0.125]

dirs = os.listdir(pathin)
print('start')
for j in range(np.size(dirs)):
    start = time.time()
    df0, stime = p2p.ReadINCAResults(pathin, VarList,
                                     SubZone=volume, Equ=equ)

    xval = np.arange(volume[0][0], volume[0][1] + dxyz[0], dxyz[0])
    yval = np.arange(volume[1][0], volume[1][1] + dxyz[1], dxyz[1])
    zval = np.arange(volume[2][0], volume[2][1] + dxyz[2], dxyz[2])

    df1 = df0[df0.x.isin(xval)]
    df2 = df1[df1.y.isin(yval)]
    df3 = df2[df2.z.isin(zval)]

    stime = np.around(stime, decimals=2)
    filename = "TP_data_" + str(stime)
    df3.to_hdf(pathout + filename + ".h5", 'w', format='fixed')
    end = time.time() - start
    print(dirs[j] + 'took' + str(end) + 's')

# %% save to tecplot format
# in front of the step
"""
df1 = df3.query("x<=0.0 & y>=0.0")
p2p.frame2tec3d(df1, pathout, filename + 'A', zname=1, stime=time)
p2p.tec2plt(pathout, filename + 'A')
# behind the step
df2 = df3.query("x>=0.0")
p2p.frame2tec3d(df2, pathout, filename + 'B', zname=2, stime=time)
p2p.tec2plt(pathout, filename + 'B')

end = time.time() - start
print(filename + ' took ' + str(end) + 's')
"""
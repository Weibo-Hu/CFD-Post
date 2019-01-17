"""
Created on Sat Jul 4 13:30:50 2019
    This code for preprocess data for futher use.
@author: Weibo Hu
"""
# %% Load necessary module
import os
from timer import timer
import tecplot as tp
import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from DMD import DMD
from scipy.interpolate import griddata
from sparse_dmd import dmd, sparse
import types


plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 18
numsize = 15

# %% load data
InFolder = "/media/weibo/Data3/BFS_M1.7L_0505/3DSnapshots/81/"
SaveFolder = "/media/weibo/Data3/BFS_M1.7L_0505/Plot/"
path = "/media/weibo/Data3/BFS_M1.7L_0505/Plot/"
path1 = "/media/weibo/Data3/BFS_M1.7L_0505/Plot/"
path2 = "/media/weibo/Data3/BFS_M1.7L_0505/3DSnapshots/"
FileID = pd.read_csv(path2 + "ReadList.dat", sep='\t')
timepoints = np.arange(880.5, 890.0 + 0.5, 0.5)
dirs = sorted(os.listdir(InFolder))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
DataFrame = pd.read_hdf(InFolder + dirs[0])
#DataFrame['x'] = DataFrame['x'].astype(float)
#DataFrame['y'] = DataFrame['y'].astype(float)
#DataFrame['z'] = DataFrame['z'].astype(float)
#NewFrame = DataFrame.query("x>=-5.0 & x<=25.0 & y>=-3.0 & y<=5.0")
#
#ind = NewFrame.index.values
xval = DataFrame['x'] # NewFrame['x']
yval = DataFrame['y']  # NewFrame['y']
zval = DataFrame['z']
#x, y = np.meshgrid(np.unique(xval), np.unique(yval))
x1 = -5.0
x2 = 25.0
y1 = -3.0
y2 = 5.0
#with timer("Load Data"):
#    Snapshots = np.vstack(
#        [pd.read_hdf(InFolder + dirs[i])['u'] for i in range(np.size(dirs))])
var0 = 'u'
var1 = 'v'
var2 = 'p'
col = [var0, var1, var2]
fa = 1 #/(1.7*1.7*1.4)
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame.ravel(order='F')
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(InFolder + dirs[i+1])
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame.ravel(order='F')))
        DataFrame += TempFrame
Snapshots = Snapshots.T

m, n = np.shape(Snapshots)
o = np.size(col)
if (m % o != 0):
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m/o)
AveFlow = DataFrame/np.size(dirs)
meanflow = AveFlow

# %% DMD 
varset = { var0: [0, m],
           var1: [m, 2*m],
           var2: [2*m, 3*m]
        }
Snapshots1 = Snapshots[:, :-1]
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
eigval = bfs.eigval

# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, bfs, dt=dt)
gamma = [700, 800, 850, 900]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)

# %% 
sp = 0
bfs1.sparse.Nz[sp]
bfs1.sparse.gamma[sp] 
r = np.size(eigval)
sp_ind = np.arange(r)[bfs1.sparse.nonzero[:, sp]]

# %% Eigvalue Spectrum
var = var0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-', fill=False,
                         label='unit circle', linewidth=7.0, alpha=0.5)
ax1.add_artist(unit_circle)
ax1.scatter(eigval.real, eigval.imag, marker='o',
            facecolor='none', edgecolors='k', s=18)
sp_eigval = eigval[sp_ind]
ax1.scatter(sp_eigval.real, sp_eigval.imag, marker='o',
            facecolor='gray', edgecolors='gray', s=18)
limit = np.max(np.absolute(eigval))+0.1
ax1.set_xlim((-limit, limit))
ax1.set_ylim((-limit, limit))
ax1.tick_params(labelsize=numsize)
ax1.set_xlabel(r'$\Re(\mu_i)$')
ax1.set_ylabel(r'$\Im(\mu_i)$')
ax1.grid(b=True, which='both', linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path+var+'DMDEigSpectrum.svg', bbox_inches='tight')
plt.show()

# %% discard the bad DMD modes
# bfs2 = bfs
bfs2 = bfs.reduce(0.5)
phi = bfs2.modes
freq = bfs2.omega/2/np.pi
beta = bfs2.beta
coeff = bfs2.amplitudes

# %% Mode frequency specturm
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
psi = np.abs(coeff)/np.max(np.abs(coeff))
ind1 = freq > 0.0 
freq1 = freq[ind1]
psi1 = np.abs(coeff[ind1])/np.max(np.abs(coeff[ind1]))
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
# ind2 = bfs1.sparse.nonzero[:, sp] & ind1
ind2 = bfs1.sparse.nonzero[bfs2.ind, sp]
ind3 = ind2[ind1]
ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()

# %% save dataframe of reconstructing flow
path5 = "/media/weibo/Data3/BFS_M1.7L_0505/video/"
base = meanflow[col].values
base[:, 2] = meanflow['p'].values*1.7*1.7*1.4
ind = 0
num = sp_ind[ind] # ind from small to large->freq from low to high
print('The frequency is', bfs.omega[num]/2/np.pi)
phase = np.linspace(0, 2*np.pi, 32, endpoint=False)
# modeflow1 = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
#             @ bfs.Vand[num, :].reshape(1, -1)
modeflow = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
xarr = xval.values.reshape(-1, 1) # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, 'u`', 'v`', 'p`']
path2 = "/media/weibo/Data3/BFS_M1.7L_0505/3DSnapshots/"
FileID = pd.read_csv(path2 + "ReadList.dat", sep='\t')
for ii in range(np.size(phase)):
    fluc = modeflow[:, 0].reshape((m, o), order='F')
    newflow = fluc.real
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    filename = "DMD" + '{:03}'.format(ii)
    with timer('save plt of t=' + str(phase[ii])):
        p2p.mul_zone2tec(path2, filename, FileID, df, time=ii)
        p2p.mul_zone2tec_plt(path2, filename, FileID, df, time=ii)
        
#%% convert data to tecplot




#for i in range(np.shape(FileID)[0]):
#    filename = "test" + '{:04}'.format(i)
#    file = FileID.iloc[i]
#    ind1 = int(file['id1'])
#    ind2 = int(file['id2'])
#    df = DataFrame.iloc[ind1:ind2 + 1]
#    zonename = 'B' + '{:010}'.format(i)
#    num=[int(file['nx']), int(file['ny']), int(file['nz'])]
#    p2p.zone2tec(path2+"test/", filename, df, zonename, num, time=200.0)

    
#%%
"""
for ii in range(np.size(phase)):
    fluc = modeflow[:, ii].reshape((m, o), order='F')
#    fluc[:, 0] = fluc[:, 0] * mag0
#    fluc[:, 1] = fluc[:, 1] * mag1
#    fluc[:, 2] = fluc[:, 2] * mag2
    newflow = fluc.real
    outfile = 'DMD'+str(np.round(phase[ii], 2))
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    df1 = df.query("x>=0.0")
    with timer('save plt of t='+str(phase[ii])):
        p2p.frame2plt(df1, path5, outfile+'_1', 
                      time=phase[ii], zonename=1)
    df2 = df.query("x<=0.0 & y>=0")
    with timer('save plt of t='+str(phase[ii])):
        p2p.frame2plt(df2, path5, outfile+'_2', 
                      time=phase[ii], zonename=2)
"""

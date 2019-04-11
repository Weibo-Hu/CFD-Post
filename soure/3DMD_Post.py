"""
Created on Sat Jul 4 13:30:50 2019
    This code for preprocess data for futher use.
@author: Weibo Hu
"""
# %% Load necessary module
import os
from timer import timer
import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
import types
import dill


plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 18
numsize = 15

var0 = 'u'
var1 = 'v'
var2 = 'w'
var3 = 'p'
col = [var0, var1, var2, var3]

# %% load first snapshot data and obtain basic information
InFolder = "/media/weibo/Data3/BFS_M1.7L_0505/3DSnapshots/snapshots1/"
path = "/media/weibo/Data3/BFS_M1.7L_0505/3D_DMD/Post/"
path2 = "/media/weibo/Data3/BFS_M1.7L_0505/3D_DMD/6200-6800/"
timepoints = np.arange(900.0, 1149.5 + 0.5, 0.5)
dirs = sorted(os.listdir(InFolder))
# if np.size(dirs) != np.size(timepoints):
#     sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# obtain the basic information of the snapshots
DataFrame = pd.read_hdf(InFolder + dirs[0])
xval = DataFrame['x']  # NewFrame['x']
yval = DataFrame['y']   # NewFrame['y']
zval = DataFrame['z']
x1 = -5.0
x2 = 25.0
y1 = -3.0
y2 = 5.0
m = np.size(xval)
n = np.size(timepoints)
o = np.size(col)
varset = {var0: [0, m],
          var1: [m, 2*m],
          var2: [2*m, 3*m],
          var3: [3 * m, 4 * m]}

# %% Load & rename data 
eigval = np.load(path2 + 'eigval.npy')  # bfs.eigval
omega = np.load(path2 + 'omega.npy')  # bfs.omega
#modes = np.load(path2 + 'modes.npy')  # bfs.omega
amplitudes = np.load(path2 + 'amplitudes.npy')  # bfs.omega
sparse = np.load(path2 + 'sparse.npz')
nonzero = sparse['nonzero']
# %% discard the bad DMD modes
# re_phi = np.load(path2 + 'Re_modes.npy')  # bfs2.modes
re_freq = np.load(path2 + 'Re_freq.npy')  # bfs2.omega/2/np.pi
re_beta = np.load(path2 + 'Re_beta.npy')  # bfs2.beta
re_coeff = np.load(path2 + 'Re_amplitudes.npy')  # bfs2.amplitudes
re_index = np.load(path2 + 'Re_index.npy')  # bfs2.ind
meanflow = pd.read_hdf(path2 + 'Meanflow.h5')

# %% Eigvalue Spectrum
var = var0
sp = 3
r = np.size(eigval)
sp_ind = np.arange(r)[nonzero[:, sp]]
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
# plt.show()

# %% Mode frequency specturm
reduction = 0
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
if reduction == 0:
    freq = omega/2/np.pi
    psi = np.abs(amplitudes)/np.max(np.abs(amplitudes))
    ind1 = freq > 0.0
    freq1 = freq[ind1]
    psi1 = np.abs(amplitudes[ind1])/np.max(np.abs(amplitudes[ind1]))
    ind2 = nonzero[:, sp]
else:
    psi = np.abs(re_coeff)/np.max(np.abs(re_coeff))
    ind1 = re_freq > 0.0  # freq > 0.0
    freq1 = re_freq[ind1]  # freq[ind1]
    psi1 = np.abs(re_coeff[ind1])/np.max(np.abs(re_coeff[ind1]))
    ind2 = nonzero[re_index, sp]
ind3 = ind2[ind1]
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
# plt.show()
print("the selected frequency:", freq[sparse['nonzero'][:, sp]])
# %% save dataframe of reconstructing flow
base = meanflow[col].values
base[:, 3] = meanflow['p'].values*1.7*1.7*1.4
# ind = 0 
num = np.where(np.round(freq, 4) == 0.0881)
print('The frequency is', freq[num])
phase = np.linspace(0, 2*np.pi, 32, endpoint=False)
# modeflow1 = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
#             @ bfs.Vand[num, :].reshape(1, -1)
modeflow = modes[:, num].reshape(-1, 1) * amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
xarr = xval.values.reshape(-1, 1)  # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, var3, 'u`', 'v`', 'w`', 'p`']
path3 = "/media/weibo/Data3/BFS_M1.7L_0505/3D_DMD/plt/"
FileID = pd.read_csv(path2 + "1ReadList.dat", sep='\t')
for ii in range(np.size(phase)):
    fluc = modeflow[:, ii].reshape((m, o), order='F')
    newflow = fluc.real
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    filename = str(np.round(freq[num], 3)) + "DMD" + '{:03}'.format(ii)
    with timer('save plt of t=' + str(phase[ii])):
        p2p.mul_zone2tec(path3, filename, FileID, df, time=ii)
        p2p.tec2plt(path3, filename, filename)

# %% convert data to tecplot
# for i in range(np.shape(FileID)[0]):
#     filename = "test" + '{:04}'.format(i)
#     file = FileID.iloc[i]
#     ind1 = int(file['id1'])
#     ind2 = int(file['id2'])
#     df = DataFrame.iloc[ind1:ind2 + 1]
#     zonename = 'B' + '{:010}'.format(i)
#     num=[int(file['nx']), int(file['ny']), int(file['nz'])]
#     p2p.zone2tec(path2+"test/", filename, df, zonename, num, time=200.0)


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
# from DMD import DMD
from sparse_dmd import dmd, sparse
# import types
# import dill


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
var4 = 'T'
col = [var0, var1, var2, var3, var4]

# %% load first snapshot data
path = "/home/scratch/whu/BFS_M1.7TS/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathD = path + "Domain/"
pathPOD = path + "POD/"
path3D = path + "3D_DMD/"
pathH = path + "hdf5/"
timepoints = np.arange(550.0, 849.5 + 0.5, 0.5)
dirs = sorted(os.listdir(pathH))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# obtain the basic information of the snapshots
DataFrame = pd.read_hdf(pathH + dirs[0])

DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += -3.0
grouped = DataFrame.groupby(['x', 'y', 'z'])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-25.0 & x<=15.0 & walldist>=0.0 & y<=8.0")
ind = NewFrame.index.values
xval = DataFrame['x'][ind] # NewFrame['x']
yval = DataFrame['y'][ind] # NewFrame['y']
zval = DataFrame['z'][ind]

x1 = -25.0
x2 = 15.0
y1 = 0.0
y2 = 8.0
m = np.size(xval)
n = np.size(timepoints)
o = np.size(col)
varset = {var0: [0, m],
          var1: [m, 2*m],
          var2: [2*m, 3*m],
          var3: [3 * m, 4 * m],
          var4: [4 * m, 5 * m]}

"""
------------------------------------------------------------------------------
structure of snapshots
t1  t2  ... tn
u1  u2  ... un
v1  v2  ... vn
w1  w2  ... wn
------------------------------------------------------------------------------
"""

# %% Load all the snapshots
print("loading data")
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame[ind].ravel(order='F')
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(pathH + dirs[i + 1])
        grouped = TempFrame.groupby(['x', 'y', 'z'])
        TempFrame = grouped.mean().reset_index()
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame[ind].ravel(order='F')))
        DataFrame += TempFrame
Snapshots = Snapshots.T
# obtain the dimensional information:
# m-No of coordinates, n-No of snapshots, o-No of variables
m, n = np.shape(Snapshots)
o = np.size(col)
if (m % o != 0):
    sys.exit("Dimensions of snapshots are wrong!!!")
m = int(m/o)
# obtain mean flow 
meanflow = DataFrame/np.size(dirs)
del TempFrame, DataFrame
print("data loaded")

# %%##########################################################################
"""
    Compute
"""
# %% DMD 
dt = 0.5
bfs = dmd.DMD(Snapshots, dt=dt)
with timer("DMD computing"):
    bfs.compute()
print("The residuals of DMD is ", bfs.residuals)
#     bfs0 = dill.load(f)
# %% Save results for plotting
meanflow.to_hdf(path3D + 'Meanflow.h5', 'w', format='fixed')
np.save(path3D + 'eigval', bfs.eigval)
tno = np.size(timepoints)
ind = int(tno / 4)
np.save(path3D + 'modes1', bfs.modes[:,:ind])
np.save(path3D + 'modes2', bfs.modes[:,ind:2*ind])
np.save(path3D + 'modes3', bfs.modes[:,2*ind:3*ind])
np.save(path3D + 'modes4', bfs.modes[:,3*ind:])
np.save(path3D + 'omega', bfs.omega)
np.save(path3D + 'beta', bfs.beta)
np.save(path3D + 'amplitudes', bfs.amplitudes)
# %% SPDMD
bfs1 = sparse.SparseDMD(Snapshots, dt=dt)
gamma = [500, 700, 900]
with timer("SPDMD computing"):
    bfs1.compute_sparse(gamma)
print("The nonzero amplitudes of each gamma:", bfs1.sparse.Nz)
nonzero = bfs1.sparse.nonzero
# %% Discard the bad modes
bfs2 = bfs.reduce(0.998)
re_freq = bfs2.omega/2/np.pi
re_beta = bfs2.beta
re_coeff = bfs2.amplitudes

# with open(path2+"dmd.bin", "wb") as f:  # save object to file 
#     dill.dump(bfs, f)
# with open(path2+"bfs.bin", "rb") as f:  # load object

# discard the bad DMD modes
percent = 0.998
bfs2 = bfs.reduce(percent)  # 0.95
np.save(path3D + 'Re_modes', bfs2.modes)
np.save(path3D + 'Re_freq', bfs2.omega/2/np.pi)
np.save(path3D + 'Re_beta', bfs2.beta)
np.save(path3D + 'Re_amplitudes', bfs2.amplitudes)
np.save(path3D + 'Re_index', bfs2.ind)

np.savez(path3D + 'sparse.npz',
         Nz=bfs1.sparse.Nz,
         gamma=bfs1.sparse.gamma,
         nonzero=bfs1.sparse.nonzero)

# %% Eigvalue Spectrum
eigval = bfs.eigval
var = var0
sp = 0
r = np.size(eigval)
sp_ind = np.arange(r)[nonzero[:, sp]]
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-',
                         fill=False, label='unit circle',
                         linewidth=7.0, alpha=0.5)
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
plt.savefig(path3D+var+'DMDEigSpectrum.svg', bbox_inches='tight')
# plt.show()

# %% Mode frequency specturm
re_index = bfs2.ind
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
psi = np.abs(re_coeff)/np.max(np.abs(re_coeff))
ind1 = re_freq > 0.0
freq1 = re_freq[ind1]
psi1 = np.abs(re_coeff[ind1])/np.max(np.abs(re_coeff[ind1]))
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
ind2 = nonzero[re_index, sp]
ind3 = ind2[ind1]
ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
            facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.savefig(path+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
# plt.show()

# %% save dataframe of reconstructing flow
"""
base = meanflow[col].values
base[:, 3] = meanflow['p'].values*1.7*1.7*1.4
ind = 0
num = np.where(np.round(omega/2/np.pi, 4) == 0.2119)
print('The frequency is', omega[num]/2/np.pi)
phase = np.linspace(0, 2*np.pi, 32, endpoint=False)
# modeflow1 = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
#             @ bfs.Vand[num, :].reshape(1, -1)
modeflow = modes[:, num].reshape(-1, 1) * amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
xarr = xval.values.reshape(-1, 1)  # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, var3, 'u`', 'v`', 'w`', 'p`']
path3 = "/home/scratch/whu/3D_DMD/plt/"
FileID = pd.read_csv(path2 + "1ReadList.dat", sep='\t')
for ii in range(np.size(phase)):
    fluc = modeflow[:, ii].reshape((m, o), order='F')
    newflow = fluc.real
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    filename = "DMD" + '{:03}'.format(ii)
    with timer('save plt of t=' + str(phase[ii])):
        p2p.mul_zone2tec(path3, filename, FileID, df, time=ii)
        p2p.tec2plt(path3, filename, filename)

"""        
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


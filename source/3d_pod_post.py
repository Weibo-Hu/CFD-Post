"""
Created on Sat Jan 10 13:30:50 2020
    Analysis of 3 dimensional proper orthogonal decomposition 
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
import pod as pod


plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 13
numsize = 10

var0 = 'u'
var1 = 'v'
var2 = 'w'
var3 = 'p'
col = [var0, var1, var2, var3]

# %% load first snapshot data
path = "/media/weibo/VID2/BFS_M1.7TS_LA/"
p2p.create_folder(path)
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathD = path + "Domain/"
pathPOD = path + "POD/"
timepoints = np.arange(1000, 1299.5 + 0.5, 0.5)
dirs = sorted(os.listdir(pathD))
if np.size(dirs) != np.size(timepoints):
    sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# obtain the basic information of the snapshots
DataFrame = pd.read_hdf(pathD + dirs[0])
xval = DataFrame['x']  # [-10.0, 30.0]
yval = DataFrame['y']  # [-3.0, 2.0]
zval = DataFrame['z']  # [-8.0, 8.0]
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
FirstFrame = DataFrame[col].values
Snapshots = FirstFrame.ravel(order='F')
with timer("Load Data"):
    for i in range(np.size(dirs)-1):
        TempFrame = pd.read_hdf(pathD + dirs[i+1])
        if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
            sys.exit('The input snapshots does not match!!!')
        NextFrame = TempFrame[col].values
        Snapshots = np.vstack((Snapshots, NextFrame.ravel(order='F')))
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

# %% POD
dt = 0.5
with timer("POD computing"):
    eigval, eigvec, phi, coeff = \
        pod.pod(Snapshots, pathF, fluc=True, method='svd')
        
meanflow.to_hdf(pathPOD + 'Meanflow.h5', 'w', format='fixed')
np.save(pathPOD + 'eigval', eigval)
np.save(pathPOD + 'eigvec', eigvec)
np.save(pathPOD + 'phi1', phi[:, :300])
np.save(pathPOD + 'phi2', phi[:, 300:])
np.save(pathPOD + 'coeff', coeff)

# %% Eigvalue Spectrum
EFrac, ECumu, N_modes = pod.pod_eigspectrum(80, eigval)
np.savetxt(path+'EnergyFraction600.dat', EFrac, fmt='%1.7e', delimiter='\t')
var = var0
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(3.3, 3.0))
xaxis = np.arange(0, N_modes + 1)
ax1.scatter(
    xaxis[1:],
    EFrac[:N_modes],
    c='black',
    marker='o',
    s=10.0,
)   # fraction energy of every eigval mode
# ax1.legend('E_i')
ax1.set_ylim(bottom=0)
ax1.set_xlabel('Mode', fontsize=textsize)
ax1.set_ylabel(r'$E_i$', fontsize=textsize)
ax1.grid(b=True, which='both', linestyle=':')
ax1.tick_params(labelsize=numsize)
ax2 = ax1.twinx()   # cumulation energy of first several modes
# ax2.fill_between(xaxis, ECumu[:N_modes], color='grey', alpha=0.5)
ESum = np.zeros(N_modes+1)
ESum[1:] = ECumu[:N_modes]
ax2.plot(xaxis, ESum, color='grey', label=r'$ES_i$')
ax2.set_ylim([0, 100])
ax2.set_ylabel(r'$ES_i$', fontsize=textsize)
ax2.tick_params(labelsize=numsize)
plt.tight_layout(pad=0.5, w_pad=0.2, h_pad=1)
plt.savefig(pathF+str(N_modes)+'_PODEigSpectrum80.svg', bbox_inches='tight')
plt.show()

# %% collect data
base = meanflow[col]
fa = 1.7 * 1.7 * 1.4
var = col[3]
base.assign(var=meanflow[var] * fa)
xarr = xval.values.reshape(-1, 1)  # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, var3, 'u`', 'v`', 'w`', 'p`']

# %% save dataframe of mode flow into tecplot
mode_id = np.arange(10, 50, 1)
for ind in mode_id:
    # ind = 10
    modeflow = phi[:, ind] * coeff[ind, 0]
    newflow = modeflow.reshape((m, o), order='F')

    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    filename = "POD_Mode" + str(ind)
    with timer("save plt"):
        df1 = df.query("x<=0.0 & y>=0.0")
        filename1 = filename + "A"
        p2p.frame2tec3d(df1, pathPOD, filename1, zname=1, stime=ind)
        p2p.tec2plt(pathPOD, filename1, filename1)
        df2 = df.query("x>=0.0")
        filename2 = filename + "B"
        p2p.frame2tec3d(df2, pathPOD, filename2, zname=2, stime=ind)
        p2p.tec2plt(pathPOD, filename2, filename2)
    
    
# %%############################################################################
"""
    POD convergence
"""
# %% POD convergence
enum = [200, 300, 400, 500, 600]
data = np.zeros((5, 4))
ener = np.loadtxt(path + 'EnergyFraction200.dat')
data[:, 0] = ener[:4]

fig, ax = plt.subplots(figsize=(3.2, 3))
ax.semilogy(enum, data[0, :] / 100, marker="o", color="k", linewidth=1.0)
ax.semilogy(enum, data[1, :] / 100, marker="^", color="k", linewidth=1.0)
ax.semilogy(enum, data[2, :] / 100, marker="*", color="k", linewidth=1.0)
ax.semilogy(enum, data[3, :] / 100, marker="s", color="k", linewidth=1.0)
lab = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
ax.legend(lab, ncol=2, loc="upper right", fontsize=15)
#          bbox_to_anchor=(1., 1.12), borderaxespad=0., frameon=False)
ax.set_ylim([0.01, 1.0])
ax.set_xlim([280, 700])
ax.set_xlabel(r"$N$", fontsize=textsize)
ax.set_ylabel(r"$\lambda_i/\sum_{k=1}^N \lambda_k$", fontsize=textsize)
ax.grid(b=True, which="both", linestyle=":")
plt.tick_params(labelsize=numsize)
plt.savefig(
    pathF + "POD/PODConvergence.svg", bbox_inches="tight", pad_inches=0.1
)
plt.show()
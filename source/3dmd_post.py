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
from scipy.interpolate import griddata
import types
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman',  # 'color' : 'k',
    'weight': 'normal',
}

matplotlib.rc('font', **font)
textsize = 13
numsize = 10

var0 = 'u'
var1 = 'v'
var2 = 'w'
var3 = 'p'
var4 = 'T'
col = [var0, var1, var2, var3, var4]

# %% load first snapshot data and obtain basic information
path = "/media/weibo/IM1/BFS_M1.7Tur/"
path3D = path + '3D_DMD/'
pathH = path + 'hdf5/'
timepoints = np.arange(951.0, 1350.5 + 0.5, 0.5)
dirs = sorted(os.listdir(pathH))
# if np.size(dirs) != np.size(timepoints):
#     sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# obtain the basic information of the snapshots
DataFrame = pd.read_hdf(pathH + dirs[0])
DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += 3.0
grouped = DataFrame.groupby(['x', 'y', 'z'])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-5.0 & x<=20.0 & walldist>=0.0 & y<=5.0")
ind = NewFrame.index.values
xval = DataFrame['x'][ind]  # NewFrame['x']
yval = DataFrame['y'][ind]   # NewFrame['y']
zval = DataFrame['z'][ind]
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
          var3: [3 * m, 4 * m],
          var4: [4 * m, 5 * m]}

# %% Load & rename data
eigval = np.load(path3D + 'eigval.npy')  # bfs.eigval
omega = np.load(path3D + 'omega.npy')  # bfs.omega
modes1 = np.load(path3D + 'modes1.npy')  # bfs.omega
modes2 = np.load(path3D + 'modes2.npy')
modes3 = np.load(path3D + 'modes3.npy')
modes4 = np.load(path3D + 'modes4.npy')
# %% merge modes
modes = np.hstack((modes1, modes2, modes3, modes4))
del modes1, modes2, modes3, modes4
amplitudes = np.load(path3D + 'amplitudes.npy')  # bfs.omega
sparse = np.load(path3D + 'sparse.npz')
nonzero = sparse['nonzero']
# %% discard the bad DMD modes
# re_phi = np.load(path2 + 'Re_modes.npy')  # bfs2.modes
re_freq = np.load(path3D + 'Re_freq.npy')  # bfs2.omega/2/np.pi
re_beta = np.load(path3D + 'Re_beta.npy')  # bfs2.beta
re_coeff = np.load(path3D + 'Re_amplitudes.npy')  # bfs2.amplitudes
re_index = np.load(path3D + 'Re_index.npy')  # bfs2.ind
# %% load mean flow
meanflow = pd.read_hdf(path3D + 'Meanflow.h5')
meanflow['walldist'] = meanflow['y']
meanflow.loc[meanflow['x'] >= 0.0, 'walldist'] += 3.0
grouped = meanflow.groupby(['x', 'y', 'z'])
meanflow = grouped.mean().reset_index()
meanflow = meanflow.query("x>=-5.0 & x<=20.0 & walldist>=0.0 & y<=5.0")
# %% spdmd selection
sp = 2
r = np.size(eigval)
sp_ind = np.arange(r)[nonzero[:, sp]]

# %% Eigvalue Spectrum
sp_ind = None
filtval = 0.99
var = var0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
unit_circle = plt.Circle((0., 0.), 1., color='grey', linestyle='-', fill=False,
                         label='unit circle', linewidth=7.0, alpha=0.5)
ax1.add_artist(unit_circle)
ind = np.where(np.abs(eigval) > filtval)
ax1.scatter(eigval.real[ind], eigval.imag[ind], marker='o',
            facecolor='none', edgecolors='k', s=18)
if sp_ind is not None:
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
plt.show()

# %% Mode frequency specturm
reduction = 0
matplotlib.rc('font', size=textsize)
fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
if reduction == 0:
    freq = omega/2/np.pi
    psi = np.abs(amplitudes)/np.max(np.abs(amplitudes))
    ind1 = (freq > 0.0) & (freq < 1.0) & (np.abs(eigval) > filtval)
    freq1 = freq[ind1]
    psi1 = np.abs(amplitudes[ind1])/np.max(np.abs(amplitudes[ind1]))
    ind2 = nonzero[:, sp]
else:
    psi = np.abs(re_coeff)/np.max(np.abs(re_coeff))
    ind1 = re_freq > 0.0  # freq > 0.0
    freq1 = re_freq[ind1]  # freq[ind1]
    psi1 = np.abs(re_coeff[ind1])/np.max(np.abs(re_coeff[ind1]))
    ind2 = nonzero[re_index, sp]
ax2.set_xscale("log")
ax2.vlines(freq1, [0], psi1, color='k', linewidth=1.0)
if sp_ind is not None:
    ind3 = ind2[ind1]
    ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
                facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_i|$')
ax2.grid(b=True, which='both', linestyle=':')
plt.savefig(path3D+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()
print("the selected frequency:", freq[sparse['nonzero'][:, sp]])

# %% save dataframe of reconstructing flow
base = meanflow[col].values
base[:, 3] = meanflow['p'].values*1.7*1.7*1.4
# ind = 0 
num = np.where(np.round(freq, 4) == 0.2068) # 0.3017
print("The frequency is", freq[num])
phase = np.linspace(0, 2*np.pi, 8, endpoint=False)
# modeflow1 = bfs.modes[:,num].reshape(-1, 1) * bfs.amplitudes[num] \
#             @ bfs.Vand[num, :].reshape(1, -1)
modeflow = modes[:, num].reshape(-1, 1) * amplitudes[num] \
           * np.exp(phase.reshape(1, -1)*1j)
xarr = xval.values.reshape(-1, 1)  # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, var3, var4,
         'u`', 'v`', 'w`', 'p`', 'T`']
path3 = path + 'plt/'
for ii in range(np.size(phase)):
    # ind = 10
    fluc = modeflow[:, ii].reshape((m, o), order='F')
    newflow = fluc.real
    data = np.hstack((xarr, yarr, zarr, base, newflow))
    df = pd.DataFrame(data, columns=names)
    filename = str(np.round(freq[num], 3)) + "DMD" + '{:03}'.format(ii)
    with timer('save plt of t=' + str(phase[ii])):
        df1 = df.query("x<=0.0 & y>=0.0")
        filename1 = filename + "A"
        p2p.frame2tec3d(df1, path3D, filename1, zname=1, stime=ii)
        p2p.tec2plt(path3D, filename1, filename1)
        df2 = df.query("x>=0.0")
        filename2 = filename + "B"
        p2p.frame2tec3d(df2, path3D, filename2, zname=2, stime=ii)
        p2p.tec2plt(path3D, filename2, filename2)

# %%      
def dmd_plt(df, path, ind):
    matplotlib.rc('font', size=textsize)   
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    ax.set_xlabel(r'$x/\delta_0$', fontdict=font)
    ax.set_ylabel(r'$y/\delta_0$', fontdict=font)

#    p = griddata((xval, yval), p_new, (x, y))
#    gradyx = np.gradient(p, y_coord, x_coord)
#    pgrady = gradyx[0]
#    pgradx = gradyx[1]
#    pgrad = np.sqrt(pgrady ** 2 + pgradx ** 2)
    xval, yval = np.meshgrid(np.unique(df['x']), np.unique(df['y']))
    corner = (xval < 0.0) & (yval < 0.0)
    u = griddata((df['x'], df['y']), df['u'] + df['u`'], (xval, yval))
    lev = np.linspace(-0.3, 1.3, 17)
    u[corner] = np.nan
    # ax.clabel(cs, inline=0, inline_spacing=numsize, fontsize=numsize-4)    
    cbar = ax.contour(xval, yval, u, levels=[0.0, 0.99],
                      colors='k', linewidths=1.0)
    cbar = ax.contourf(xval, yval, u, levels=lev, cmap='bwr_r')
    ax.set_xlim((-5.0, 20.0))
    ax.set_ylim((-3.0, 2.0))
    # Add shock wave
    # shock = np.loadtxt(pathM+'ShockLineFit.dat', skiprows=1)
    # ax.plot(shock[:, 0], shock[:, 1], 'g', linewidth=1.0)
    # ax.set_title(r'$\omega t={}/8\pi$'.format(i), fontsize=textsize)
    # add colorbar
    ax_divider = make_axes_locatable(ax)
    cbaxes = ax_divider.append_axes("top", size="7%", pad="25%")
    cbar = plt.colorbar(cbar, extendrect='False', cax=cbaxes, # ticks=rg2
                        fraction=0.25, orientation="horizontal")
    cbar.set_label(r'$|\nabla p|\delta_0/p_\infty$', rotation=0, 
                   x=-0.15, y=1.3, labelpad=-23, fontsize=textsize-1)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.ax.xaxis.offsetText.set_fontsize(numsize)
    cbar.update_ticks()
    plt.savefig(path + 'DMDAnimat'+ind+'.jpg', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% save plots for video
pathA = path + 'animation/'
fn1 = '[0.02]DMD'
fstr = '0p02'
if not os.path.exists(pathA + fstr):
    os.mkdir(pathA + fstr)
path_id = pathA + fstr + '/'
num = np.arange(16)
for ii in range(np.size(num)):
    fn2 = str(num[ii])
    fn3 = fn2.zfill(3)
    phase = num[ii] * np.pi
    fn = [pathA + fn1 + fn3 + 'A.plt', pathA + fn1 + fn3 + 'B.plt']
    df = p2p.ReadAllINCAResults(FoldPath=pathA, SpanAve=None, FileName=fn)
    df1 = df.loc[df['z'] == 0]
    dmd_plt(df1, path_id, fn3)

# %% Convert plots to animation
import imageio
from natsort import natsorted, ns
dirs = os.listdir(path_id)
dirs = natsorted(dirs, key=lambda y: y.lower())
with imageio.get_writer(path_id+fstr+'DMDAnima.mp4', mode='I', fps=2) as writer:
    for filename in dirs:
        image = imageio.imread(path_id + filename)
        writer.append_data(image)
    
# %%
# path3 = "/media/weibo/Data3/BFS_M1.7L_0505/3D_DMD/plt/"
# FileID = pd.read_csv(path2 + "1ReadList.dat", sep='\t')
# for ii in range(np.size(phase)):
#     fluc = modeflow[:, ii].reshape((m, o), order='F')
#     newflow = fluc.real
#     data = np.hstack((xarr, yarr, zarr, base, newflow))
#     df = pd.DataFrame(data, columns=names)
#     filename = str(np.round(freq[num], 3)) + "DMD" + '{:03}'.format(ii)
#     with timer('save plt of t=' + str(phase[ii])):
#         p2p.mul_zone2tec(path3, filename, FileID, df, time=ii)
#         p2p.tec2plt(path3, filename, filename)

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


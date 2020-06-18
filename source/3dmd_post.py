"""
Created on Sat Jul 4 13:30:50 2019
    This code for preprocess data for futher use.
@author: Weibo Hu
"""
# %% Load necessary module
import os
from timer import timer
import tecplot as tp
from glob import glob
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
path3D = path + '3D_DMD_1200/'
pathH = path + 'hdf5/'
pathM = path + 'MeanFlow/'
timepoints = np.arange(951.0, 1450.5 + 0.5, 0.5)
dirs = sorted(os.listdir(pathH))
# if np.size(dirs) != np.size(timepoints):
#     sys.exit("The NO of snapshots are not equal to the NO of timespoints!!!")
# obtain the basic information of the snapshots
DataFrame = pd.read_hdf(pathH + dirs[0])
DataFrame['walldist'] = DataFrame['y']
DataFrame.loc[DataFrame['x'] >= 0.0, 'walldist'] += 3.0
grouped = DataFrame.groupby(['x', 'y', 'z'])
DataFrame = grouped.mean().reset_index()
NewFrame = DataFrame.query("x>=-5.0 & x<=20.0 & walldist>=0.0 & y<=5.0") # 20
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
dt = 0.5
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
meanflow = meanflow.query("x>=-5.0 & x<=20.0 & walldist>=0.0 & y<=5.0")  # 20
# %% spdmd selection
sp = 2
r = np.size(eigval)
sp_ind = np.arange(r)[nonzero[:, sp]]

# %% Eigvalue Spectrum
sp_ind = None
filtval = 0
var = var0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rc('font', size=textsize)
fig1, ax1 = plt.subplots(figsize=(2.8, 3.0))
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
ax1.set_xlabel(r'$\Re(\mu_k)$')
ax1.set_ylabel(r'$\Im(\mu_k)$')
ax1.grid(b=True, which='both', linestyle=':')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(path3D+var+'DMDEigSpectrum.svg', bbox_inches='tight')
plt.show()

# %% Mode frequency specturm
reduction = 0
filtval = 0.94
matplotlib.rc('font', size=textsize)
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
if reduction == 0:
    freq = omega/2/np.pi
    psi = np.abs(amplitudes)/np.max(np.abs(amplitudes))
    ind1 = (freq > 0.0) & (freq < 1.0) & (np.abs(eigval) > filtval)
    freq1 = freq[ind1]
    psi1 = np.abs(amplitudes[ind1])/np.max(np.abs(amplitudes[ind1]))
    beta1 = np.real(np.log(eigval[ind1])/dt)
    ind2 = nonzero[:, sp]
else:
    psi = np.abs(re_coeff)/np.max(np.abs(re_coeff))
    ind1 = re_freq > 0.0  # freq > 0.0
    freq1 = re_freq[ind1]  # freq[ind1]
    psi1 = np.abs(re_coeff[ind1])/np.max(np.abs(re_coeff[ind1]))
    ind2 = nonzero[re_index, sp]
ax2.set_xscale("log")
colors = plt.cm.Greys_r(beta1/np.min(beta1))
ax2.vlines(freq1, [0], psi1, color=colors, linewidth=1.0)
if sp_ind is not None:
    ind3 = ind2[ind1]
    ax2.scatter(freq1[ind3], psi1[ind3], marker='o',
                facecolor='gray', edgecolors='gray', s=15)
ax2.set_ylim(bottom=0.0)
ax2.tick_params(labelsize=numsize, pad=6)
ax2.set_xlabel(r'$f \delta_0/u_\infty$')
ax2.set_ylabel(r'$|\psi_k|$')
ax2.grid(b=True, which='both', linestyle=':', alpha=0.0)
plt.savefig(path3D+var+'DMDFreqSpectrum.svg', bbox_inches='tight')
plt.show()
print("the selected frequency:", freq[sparse['nonzero'][:, sp]])

savenm = ['freq', 'psi', 'beta']
savevr = np.vstack((freq1, psi1, beta1))
frame = pd.DataFrame(data=savevr.T, columns=savenm)
frame.sort_values(by=['freq'], inplace=True)
frame.to_csv(path3D + 'ModeInfo.dat', index=False,
             float_format='%1.8e', sep=' ')
# %% save dataframe of reconstructing flow
base = meanflow[col].values
base[:, 3] = meanflow['p'].values*1.7*1.7*1.4
# ind = 0 
num = np.where(np.round(freq, 4) == 0.2033) # 0.3017 # 0.08224 # 0.2033 # 0.2134
print("The frequency is", freq[num])
phase = np.linspace(0, 2*np.pi, 4, endpoint=False)
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
    filename = filename.replace(".", "p")
    with timer('save plt of t=' + str(phase[ii])):
        df1 = df.query("x<=0.0 & y>=0.0")
        filename1 = filename + "A"
        p2p.frame2tec3d(df1, path3D, filename1, zname=1, stime=ii)
        p2p.tec2plt(path3D, filename1, filename1)
        df2 = df.query("x>=0.0")
        filename2 = filename + "B"
        p2p.frame2tec3d(df2, path3D, filename2, zname=2, stime=ii)
        p2p.tec2plt(path3D, filename2, filename2)

# %% dat file to plt file
dirs = os.listdir(path3D + '0p085/')
for jj in range(np.size(dirs)):
    filename = os.path.splitext(dirs[jj])[0]
    print(filename)
    p2p.tec2plt(path3D + '0p085/', filename, filename)

"""
plot 2D slice contour 

"""
# %% generate data
num = np.where(np.round(freq, 4) == 0.0229) # 0.3017
base = meanflow[col].values
base[:, 3] = meanflow['p'].values*1.7*1.7*1.4
print("The frequency is", freq[num])
phase = 0.5 * np.pi
modeflow = modes[:, num].reshape(-1, 1) * amplitudes[num] * np.exp(phase)
fluc = modeflow.reshape((m, o), order='F')
newflow = fluc.real
xarr = xval.values.reshape(-1, 1)  # row to column
yarr = yval.values.reshape(-1, 1)
zarr = zval.values.reshape(-1, 1)
names = ['x', 'y', 'z', var0, var1, var2, var3, var4,
         'u`', 'v`', 'w`', 'p`', 'T`']
data = np.hstack((xarr, yarr, zarr, base, newflow))
df = pd.DataFrame(data, columns=names)

# %% load data
freq1 = 0.0755  # freq[num]
path1 = path3D + '0p0755/'
files = glob(path1 + '*DMD000?.plt')
df = p2p.ReadAllINCAResults(path1, FileName=files)
# %% in X-Y plane, preprocess
var = 'u'
avg = True
amp = 1.0  # for fluctuations
fa = 0.0   # for mean value
sliceflow = df.loc[df['z']==0]
if var == 'u':
    varval = sliceflow[var] * fa + sliceflow['u`'] * amp
    grouped = df.groupby(['x', 'y'])
    df2 = grouped.mean().reset_index()
    varval = df2['u`']

if var == 'p':
    varval = sliceflow[var] * fa + sliceflow['p`'] * amp
    grouped = df.groupby(['x', 'y'])
    df2 = grouped.mean().reset_index()
    varval = df2['p`']

if var == 'rho':
    p_t = sliceflow['p'] * fa + sliceflow['p`'] * amp
    T_t = sliceflow['T'] * fa + sliceflow['T`'] * amp
    varval = p_t / T_t

if avg == True:
    xarr = df2['x']
    yarr = df2['y']
else:
    xarr = sliceflow['x']
    yarr = sliceflow['y']
x, y = np.meshgrid(np.unique(xarr), np.unique(yarr))
print("Limit value: ", np.min(varval), np.max(varval))
u = griddata((xarr, yarr), varval, (x, y))
corner = (x < 0.0) & (y < 0.0)
u[corner] = np.nan
# %% in X-Y plane, plot
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(6.6, 2.8))
c1 = -0.09 # -0.13 #-0.024
c2 = -c1 # 0.010 #0.018
lev1 = np.linspace(c1, c2, 21)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, y, u, levels=lev1, cmap='RdBu_r', extend='both') 
ax.set_xlim(-5.0, 20.0)
ax.set_ylim(-3.0, 2.0)
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbaxes = fig.add_axes([0.18, 0.76, 0.30, 0.07])  # x, y, width, height
cbar1 = plt.colorbar(cbar, cax=cbaxes, orientation="horizontal",
                     extendrect='False', ticks=rg2)
cbar1.formatter.set_powerlimits((-2, 2))
cbar1.ax.xaxis.offsetText.set_fontsize(numsize)
cbar1.update_ticks()
cbar1.set_label(r'$\Re(\phi_{})$'.format(var), rotation=0, 
                x=1.16, labelpad=-26, fontsize=textsize)
cbaxes.tick_params(labelsize=numsize)
# add shock wave
shock = np.loadtxt(pathM+'ShockLineFit.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], color='#32cd32ff', linewidth=1.2)
# add streamline1
shock = np.loadtxt(pathM+'streamline1.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], linestyle='--', color='#32cd32ff', linewidth=1.2)
# add streamline1
shock = np.loadtxt(pathM+'streamline3.dat', skiprows=1)
ax.plot(shock[:, 0], shock[:, 1], linestyle='--', color='#32cd32ff', linewidth=1.2)
# Add sonic line
# sonic = np.loadtxt(pathM+'SonicLine.dat', skiprows=1)
# ax.plot(sonic[:, 0], sonic[:, 1], color='#32cd32ff',
#        linestyle='--', linewidth=1.5)
# add boundary layer
# boundary = np.loadtxt(pathM+'BoundaryEdge.dat', skiprows=1)
# ax.plot(boundary[:, 0], boundary[:, 1], 'k', linewidth=1.2)
# Add dividing line(separation line)
dividing = np.loadtxt(pathM+'BubbleLine.dat', skiprows=1)
ax.plot(dividing[:, 0], dividing[:, 1], 'k--', linewidth=1.2)
# ax.annotate("(a)", xy=(-0.1, 1.), xycoords='axes fraction', fontsize=textsize)
filename = path3D + var + str(np.round(freq1, 3)) + 'DMDModeXY_avg.svg'
plt.savefig(filename, bbox_inches='tight')
plt.show()

# %% plot in X-Z
"""
plot 2D slice contour in X-Z plane

"""
var = 'u'
amp = 1.0  # for fluctuations
fa = 1.0   # for mean value
sliceflow = df.loc[df['y']==-2.875]
xarr = sliceflow['x']
zarr = sliceflow['z']
x, z = np.meshgrid(np.unique(xarr), np.unique(zarr))
if var == 'u':
    varval = sliceflow[var] * fa + sliceflow['u`'] * amp

if var == 'p':
    varval = sliceflow[var] * fa + sliceflow['p`'] * amp

if var == 'rho':
    varval = (sliceflow['p'] * fa + sliceflow['p`'] * amp) / \
             (sliceflow['T'] * fa + sliceflow['T`'] * amp) 

print("Limit value: ", np.min(varval), np.max(varval))
u = griddata((xarr, zarr), varval, (x, z))
#corner = (x < 0.0) & (y < 0.0)
#u[corner] = np.nan
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(6.6, 2.8))
c1 = -1.0 #-0.024
c2 = 1.3 # -c1 # 0.010 #0.018
lev1 = np.linspace(c1, c2, 21)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(x, z, u, levels=lev1, cmap='jet', extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(0.0, 20.0)
ax.set_yticks(np.linspace(-8.0, 8.0, 5))
ax.tick_params(labelsize=numsize)
cbar.cmap.set_under('#053061')
cbar.cmap.set_over('#67001f')
ax.set_xlabel(r'$x/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$z/\delta_0$', fontsize=textsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True, fraction=0.016, pad=0.03)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(
    r'$\Re(\phi_{})$'.format(var), rotation=0,
    fontsize=numsize, labelpad=-26, y=1.12
)
ax.axvline(x=8.9, color="k", linestyle="--", linewidth=1.2)
filename = path3D + var + str(np.round(freq1, 3)) + 'DMDModeXZ.svg'
plt.savefig(filename, bbox_inches='tight')
plt.show()


# %% plot in X-Z
"""
plot 2D slice contour in Z-Y plane
"""
# %% load data
freq1 = 0.022  # freq[num]
path1 = path3D + '0p022/'
files = glob(path1 + '*DMD009?.plt')
equs = ['{dudx}=ddx({u`})',
        '{dvdx}=ddx({v`})',
        '{dwdx}=ddx({w`})',
        '{dudy}=ddy({u`})',
        '{dvdy}=ddy({v`})',
        '{dwdy}=ddy({w`})',
        '{dudz}=ddz({u`})',
        '{dvdz}=ddz({v`})',
        '{dwdz}=ddz({w`})',
        '{vorticity_x}={dwdy}-{dvdz}',
        '{vorticity_y}={dudz}-{dwdx}',
        '{vorticity_z}={dvdx}-{dudy}'
        ]
df = p2p.ReadAllINCAResults(path1, FileName=files, Equ=equs)

# %% plot
var = 'vorticity_x'
fa = 1.0   # for mean value
amp = 1.0  # for fluctuations
sliceflow = df.loc[df['x']==10.0]
zarr = sliceflow['z']
yarr = sliceflow['y']
z, y = np.meshgrid(np.unique(zarr), np.unique(yarr))
varval = sliceflow[var]
w_t = sliceflow['w'] * fa + sliceflow['w`'] * amp
v_t = sliceflow['v'] * fa + sliceflow['v`'] * amp
print("Limit value: ", np.min(varval), np.max(varval))
vor = griddata((zarr, yarr), varval, (z, y))
#corner = (x < 0.0) & (y < 0.0)
#u[corner] = np.nan
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(6.6, 2.6))
c1 = -5.0 #-0.024
c2 = 7.0 # -c1 # 0.010 #0.018
lev1 = np.linspace(c1, c2, 31)
lev2 = np.linspace(c1, c2, 6)
cbar = ax.contourf(z, y, vor, levels=lev1, cmap='PRGn_r', extend='both') 
#cbar = ax.contourf(x, y, u,
#                   colors=('#66ccff', '#e6e6e6', '#ff4d4d'))  # blue, grey, red
ax.set_xlim(0.5, -3.5)
ax.set_ylim(-3.0, -1.5)
ax.set_yticks(np.linspace(-3.0, -1.5, 4))
ax.tick_params(labelsize=numsize)
#cbar.cmap.set_under('#053061')
#cbar.cmap.set_over('#67001f')
ax.set_aspect('equal')
ax.set_xlabel(r'$z/\delta_0$', fontsize=textsize)
ax.set_ylabel(r'$y/\delta_0$', fontsize=textsize)
# add colorbar
rg2 = np.linspace(c1, c2, 3)
cbar = plt.colorbar(cbar, ticks=rg2, extendrect=True,
                    fraction=0.03, pad=0.03,
                    shrink=0.74, aspect=10)
cbar.ax.tick_params(labelsize=numsize)
cbar.set_label(
    r'$\phi(\omega_x)$'.format(var), rotation=0,
    fontsize=textsize-1, labelpad=-21, y=1.14
)
# add streamlines
w = griddata((sliceflow.z, sliceflow.y), sliceflow['w`'], (z, y))
v = griddata((sliceflow.z, sliceflow.y), sliceflow['v`'], (z, y))
# x, y must be equal spaced
ax.streamplot(
    z,
    y,
    w,
    v,
    density=[8, 4],
    color="k",
    arrowsize=1.2,
    linewidth=0.6,
    integration_direction="both",
)
ax.annotate("(b)", xy=(-0.13, 0.97), xycoords='axes fraction', fontsize=numsize+1)
filename = path3D + var + str(np.round(freq1, 3)) + 'DMDModeZY_9.svg'
plt.savefig(filename, bbox_inches='tight')
plt.show()

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


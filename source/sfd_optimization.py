"""
Created on Sat Jul 4 13:30:50 2019
    This code for finding the optimized parameters from DMD for SFD.
@author: Weibo Hu
"""
# %% Load necessary module
from timer import timer
import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt


# %% figures properties settings
plt.close("All")
plt.rc("text", usetex=True)
font = {
    "family": "Times New Roman",  # 'color' : 'k',
    "weight": "normal",
    "size": "large",
}
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
textsize = 18
numsize = 15

# %% data path settings
path = "/media/weibo/Data3/BFS_M1.7L_0505/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + 'video/'
pathD = path + 'DMD/'

# %% load data and test if SFD is suitable
u_inf = np.sqrt(1.4 * 287 * 190.1) * 1.7
delta_0 = 1e-3
dt = 0.5  # * delta_0 / u_inf
data = np.load(pathD + 'eigval.npy')
# remove the modes not along the unit circle
ind = np.where(np.abs(data) > 0.999)[0]
eig_reduce = data[ind]
lamb_reduce = np.log(eig_reduce) / dt
name = ['mu_r', 'mu_i', 'r', 'i']
val = np.vstack((eig_reduce.real, eig_reduce.imag,
                 lamb_reduce.real, lamb_reduce.imag))
df = pd.DataFrame(data=val.T, columns=name)
temp = df.query("r >= 0.0 & i > 0.0") # if lamb_r > 0 ???
temp = temp.sort_values(by=['i'])
eigval = temp[['mu_r', 'mu_i']]
lamb = temp[['r', 'i']]
eig_complex = eigval['mu_r'].values + 1j * eigval['mu_i'].values
lamb_complex = lamb['r'].values + 1j * lamb['i'].values
crit = np.abs(eig_complex)**2/eig_complex.imag
if np.max(eigval['mu_i']) > np.min(crit):
    sys.exit('SFD will not stablize the flow field!!!')
# save estimation results of sfd parameters
chi = (np.abs(eig_complex) + eig_complex.imag) / 2.0
delta = 2.0 / (np.abs(eig_complex) - eig_complex.imag)
freq = eig_complex.real / 2 / np.pi
sfd = np.column_stack((freq, eig_complex.real, eig_complex.imag, chi, delta))
varname = 'frequency, mu_r, mu_i, chi, delta'
np.savetxt(pathD + 'sfd_parameters.dat', sfd, header=varname,
           fmt='%8e', delimiter='\t')
del chi, delta, sfd

# %% create optimization range
coeff = np.arange(0.0, 2.0 + 0.1, 0.1)
width = np.arange(1.0, 100 + 2, 2)
chi, delta = np.meshgrid(coeff, width)

# %% spectral radius function
def sfd_radius(chi, delta, lamb, dt):
    F = lamb - chi - 1 / delta
    sigma1 = F + np.sqrt(F ** 2 / 4 + lamb / delta)
    lambda_sfd = lamb - chi * (1 - sigma1)
    temp = np.exp(lambda_sfd * dt)
    radius = np.abs(temp)
    return (radius)

# %%
tol = 0.9
it = np.shape(lamb_complex)[0]
f = open(pathD + 'sfd_optimization.dat', 'w')
f.write('# lambda_r, lambda_i, chi, delta, radius \n')
for k in range(it):
    val = lamb_complex[k]
    radius = sfd_radius(chi, delta, val, dt)

    ind = np.unravel_index(radius.argmin(), radius.shape)
    chi_temp = chi[ind]
    delta_temp = delta[ind]
    val_temp = np.delete(lamb_complex, k)
    restrict = sfd_radius(chi_temp, delta_temp, val_temp, dt)
    if np.max(restrict) < tol:
        r_min = radius[ind]
        if r_min != np.min(radius):
            sys.exit("The minimum does not match!!!")
        else:
            arr = np.array([val.real, val.imag, chi_temp, delta_temp, r_min])
        np.savetxt(f, arr.reshape(1, 5), fmt='%8e', delimiter='\t')
        # f.write(str(arr))
        # plot the contour of chi and delta
        matplotlib.rc('font', size=textsize)
        fig, ax = plt.subplots(figsize=(6.4, 6))
        cbar = ax.contourf(chi, delta, radius, cmap='rainbow')
        ax.set_xlabel(r'$\chi$', fontdict=font)
        ax.set_ylabel(r'$\Delta$', fontdict=font)
        ax.tick_params(labelsize=numsize)
        cbar1 = plt.colorbar(cbar, orientation='vertical')
        plt.savefig(pathD + 'sfd' + str(k) + '.svg')
        plt.show()
        plt.close()

f.close()

# %%
matplotlib.rc('font', size=textsize)
fig, ax = plt.subplots(figsize=(6.4, 6))
lev = np.linspace(0.95, 1.02, 21)
cbar = ax.contourf(chi, delta, radius, levels=lev, cmap='rainbow')
ax.set_xlabel(r'$\chi$', fontdict=font)
ax.set_ylabel(r'$\Delta$', fontdict=font)
ax.tick_params(labelsize=numsize)
cbar1 = plt.colorbar(cbar, ticks=lev, orientation='vertical')
plt.savefig(pathD + 'sfd.svg')
plt.show()




# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:51:30 2022

@author: Weibo
"""
# %% import modules
import numpy as np
import matplotlib.pyplot as plt
# %% save wavy wall data
x1 = np.linspace(-1, 36, 100, endpoint=True)
x2 = np.linspace(36, 44, 5000, endpoint=True)
x3 = np.linspace(44, 81, 100, endpoint=True)
y1 = np.zeros(np.size(x1))
y2 = 0.2 * np.sin(2 * np.pi / 2 * x2 + np.pi / 2) - 0.2
y3 = np.zeros(np.size(x3))

xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
arr = np.vstack((xx, yy)).T
path = "E:/cases/wavy6/"
np.savetxt(
    path + "wavy.dat", arr, header="x, y", comments="", delimiter=",", fmt="%9.6f"
)

# %% plot wavy wall
xy = pd.read_csv(path + "WallBoundary.dat", skipinitialspace=True)
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xy["x"], xy["y"], "b-")
ax.set_xlim(10, 110)  # ((np.min(xx), np.max(xx)))
ax.grid(which="both")
# plt.savefig('wavy_wall.png')
plt.show()

# %% plot wavy wall
x1 = np.linspace(0, 6, 100, endpoint=True)
x2 = np.linspace(6, 88.811328, 2000, endpoint=True)
x3 = np.linspace(88.811328, 360, 100, endpoint=True)
y1 = np.zeros(np.size(x1))
alpha = 0.758735
lambd = 2 * np.pi / alpha
amplit = 0.2 * 2.8
y2 = amplit * np.sin(alpha * (x2 - 6) + np.pi / 2) - amplit
y3 = np.zeros(np.size(x3))
xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
arr = np.vstack((xx, yy)).T
path = "E:/cases/wavy_0804/"
np.savetxt(
    path + "wavy.dat", arr, header="x, y", comments="", delimiter=",", fmt="%9.6f"
)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xx, yy, "b-")
ax.set_xlim((np.min(xx), np.max(xx)))
ax.grid(which="both")
# plt.savefig('wavy_wall.png')
plt.show()

# %% heating/cooling region using sigmod function
xa = np.linspace(30, 75, 100, endpoint=True)
xb = np.linspace(75, 120, 100, endpoint=True)
x1 = 50
x2 = 100
xp = x2 - x1
ta1 = 8.89
ta2 = 13.33
ta3 = 5.0
ta4 = 3.33
tw = 6.67
ya1 = tw + (ta1-tw)/(1+np.exp(-20*(xa-x1-0.25*xp)/(xp)))
yb1 = tw + (ta1-tw)/(1+np.exp(20*(xb-x2+0.25*xp)/(xp)))
ya2 = tw + (ta2-tw)/(1+np.exp(-20*(xa-x1-0.25*xp)/(xp)))
yb2 = tw + (ta2-tw)/(1+np.exp(20*(xb-x2+0.25*xp)/(xp)))
ya3 = tw + (ta3-tw)/(1+np.exp(-20*(xa-x1-0.25*xp)/(xp)))
yb3 = tw + (ta3-tw)/(1+np.exp(20*(xb-x2+0.25*xp)/(xp)))
ya4 = tw + (ta4-tw)/(1+np.exp(-20*(xa-x1-0.25*xp)/(xp)))
yb4 = tw + (ta4-tw)/(1+np.exp(20*(xb-x2+0.25*xp)/(xp)))
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xa, ya1, "r:", label='H1')
ax.plot(xb, yb1, "r:")
ax.plot(xa, ya2, "r-", label='H2')
ax.plot(xb, yb2, "r-")
ax.plot(xa, ya3, "b:", label='C1')
ax.plot(xb, yb3, "b:")
ax.plot(xa, ya4, "b-", label='C2')
ax.plot(xb, yb4, "b-")
# ax.legend(labels=["H1", "H2", "C1", "C2"], ncols=2)
ax.set_xlim((np.min(xa), np.max(xb)))
ax.grid(which="both")
ax.set_xlabel(r'$x$', fontsize=16)
ax.set_ylabel(r'$T_w/T_\infty$', fontsize=16)
plt.legend()
plt.savefig('heating_wall.png')
plt.show()

# %% convert INCA results to baseflow
import source.teciolib as pt
import tec2plot3d as tp3
path = '/mnt/work/cases/flat_SFD/'
path1 = path + 'TP_data_00054278/'
pt.ReadINCAResults(path, FileName=path+'Meanflow.szplt', SpanAve=True, SavePath=path, OutFile='Baseflow')
# %%
filenm = 'Baseflow_02000.00.h5'
nms = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T']
vars = ['rho', 'u', 'v', 'w', 'T', 'p']
tp3.csv2plot3d(path, filenm, vars, option='2d', skip=1)

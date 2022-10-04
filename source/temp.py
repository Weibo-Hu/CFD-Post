# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:51:30 2022

@author: Weibo
"""
# %% generate curve
import numpy as np

x1 = np.linspace(-1, 36, 100, endpoint=True)
x2 = np.linspace(36, 44, 5000, endpoint=True)
x3 = np.linspace(44, 81, 100, endpoint=True)
y1 = np.zeros(np.size(x1))
y2 = 0.2*np.sin(2*np.pi/2*x2 + np.pi/2) - 0.2
y3 = np.zeros(np.size(x3))

xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
arr = np.vstack((xx, yy)).T
path = 'E:/cases/wavy6/'
np.savetxt(path + 'wavy.dat', arr, header='x, y',
           comments='', delimiter=',', fmt='%9.6f')

# %%
x1 = np.linspace(0, 6, 100, endpoint=True)
x2 = np.linspace(6, 88.811328, 2000, endpoint=True)
x3 = np.linspace(88.811328, 360, 100, endpoint=True)
y1 = np.zeros(np.size(x1))
alpha = 0.758735
lambd = 2*np.pi/alpha
amplit = 0.2 * 2.8
y2 = amplit*np.sin(alpha*(x2-6)+np.pi/2) - amplit
y3 = np.zeros(np.size(x3))
xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
arr = np.vstack((xx, yy)).T
path = 'E:/cases/wavy_0804/'
np.savetxt(path + 'wavy.dat', arr, header='x, y',
           comments='', delimiter=',', fmt='%9.6f')

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xx, yy, 'b-')
ax.set_xlim((np.min(xx), np.max(xx)))
ax.grid(which='both')
# plt.savefig('wavy_wall.png')
plt.show()

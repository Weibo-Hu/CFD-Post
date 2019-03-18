"""
Created on Fri Aug 17 13:30:50 2018
    This code for converting 3D dataframe data to tecplot format.
    This code is only valid for a specific structure grid.
@author: Weibo Hu
"""
# %% Load necessary module
from timer import timer
import numpy as np
import plt2pandas as p2p
import pandas as pd
# %% convert h5 to szplt for time-average data
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
filename = 'MeanFlow'
# %% 0.0<= x <=20.0
# nz = 80
with timer("Read dataframe"):
    dataframe = pd.read_hdf(FoldPath + filename + '.h5')
    newframe = dataframe.query("x>=0.0 & x<=20.0 & y<=0.0")
z = np.linspace(-2.5, 2.5, 81)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'A', z=z)
# nz = 40
newframe = dataframe.query("x>=0.0 & x<=20.0 & y>=0.0 & y<=0.5")
z = np.linspace(-2.5, 2.5, 41)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'B', z=z)
# nz = 20
newframe = dataframe.query("x>=0.0 & x<=20.0 & y>=0.5 & y<=2.0")
z = np.linspace(-2.5, 2.5, 21)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'C', z=z)
# nz = 10
newframe = dataframe.query("x>=0.0 & x<=20.0 & y>=2.0 & y<=3.0")
z = np.linspace(-2.5, 2.5, 11)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'D', z=z)
# nz = 5
newframe = dataframe.query("x>=0.0 & x<=20.0 & y>=3.0 & y<=10.0")
z = np.linspace(-2.5, 2.5, 6)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'E', z=z)
# %% x>20.0
# nz = 80
dataframe['y'] = dataframe['y'].astype(float)
newframe = dataframe.query("x>=20.0 & y<=-1.5")
z = np.linspace(-2.5, 2.5, 81)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'F', z=z)
# nz = 40, x>20.0
newframe = dataframe.query("x>=20.0 & y>=-1.5 & y<=0.0")
z = np.linspace(-2.5, 2.5, 41)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'G', z=z)
# nz = 20
newframe = dataframe.query("x>=20.0 & y>=0.0 & y<=1.0")
z = np.linspace(-2.5, 2.5, 21)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'H', z=z)
# nz = 10
newframe = dataframe.query("x>=20.0 & y>=1.0 & y<=3.0")
z = np.linspace(-2.5, 2.5, 11)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'I', z=z)
# nz = 5
newframe = dataframe.query("x>=20.0 & y>=3.0& y<=10.0")
z = np.linspace(-2.5, 2.5, 6)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'J', z=z)

# %% -40.0<= x <=0.0
# nz = 80, x<0.0
newframe = dataframe.query("x<=0.0 & y>=0.0 & y<=1.0")
z = np.linspace(-2.5, 2.5, 81)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'K', z=z)
# nz = 40, x<0.0
newframe = dataframe.query("x<=0.0 & y>=1.0 & y<=1.5")
z = np.linspace(-2.5, 2.5, 41)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'L', z=z)
# nz = 20, x<0.0
newframe = dataframe.query("x<=0.0 & y>=1.5 & y<=2.0")
z = np.linspace(-2.5, 2.5, 21)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'M', z=z)
# nz = 10, x<0.0
newframe = dataframe.query("x<=0.0 & y>=2.0 & y<=3.0")
z = np.linspace(-2.5, 2.5, 11)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'N', z=z)
# nz = 5, x<0.0
newframe = dataframe.query("x<=0.0 & y>=3.0 & y<=10.0")
z = np.linspace(-2.5, 2.5, 6)
with timer("Convert .h5 to tecplot .szplt"):
    p2p.frame2szplt(newframe, FoldPath, filename + 'O', z=z)

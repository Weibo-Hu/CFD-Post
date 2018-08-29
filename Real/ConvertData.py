"""
Created on Sat Jul 4 13:30:50 2018
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

# %% Make spanwise-averaged snapshots

VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'Q-criterion',
    'L2-criterion',
    '|grad(rho)|',
    'T',
]
FoldPath = "/media/weibo/Data3/BFS_M1.7L_0505/84/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/84/"
NoBlock = 240
dirs1 = os.listdir(FoldPath)
dirs = os.scandir(FoldPath)
for folder in dirs:
    path = FoldPath+folder.name+"/"
    with timer("Read "+folder.name+" data"):
        DataFrame = p2p.NewReadINCAResults(NoBlock, path, VarList,
                                           OutFolder, SpanAve="Yes")

# %% Save time-averaged flow field
"""
VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'div', 'vorticity_1',
           'vorticity_2', 'vorticity_3', 'shear', 'Q-criterion',
           'L2-criterion', 'grad(rho)_1', 'grad(rho)_2', 'grad(rho)_3',
           '|grad(rho)|', 'Mach', 'entropy', 'T']
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/7/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
dirs = os.scandir(FoldPath)
num = np.size(os.listdir(FoldPath))
for i, folder in enumerate(dirs):
    path = FoldPath+folder.name+"/"
    if i == 0:
        with timer("Read "+folder.name+" data"):
            SumFrame = p2p.NewReadINCAResults(240, path, VarList,
                                              OutFolder, OutFile=False)
    else:
        with timer("Read "+folder.name+" data"):
            DataFrame = p2p.NewReadINCAResults(240, path, VarList,
                                               OutFolder, OutFile=False)
        if np.size(DataFrame.x) != np.size(SumFrame.x):
            sys.exit("DataFrame does not match!!!")
        else:
            SumFrame = SumFrame + DataFrame

MeanFrame = SumFrame/num

MeanFrame.to_hdf(OutFolder+"MeanFlow7.h5", 'w', format='fixed')
"""
# %% Time-average DataFrame
"""
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/Ave/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
dirs = os.scandir(FoldPath)
num = np.size(os.listdir(FoldPath))
SumFrame = pd.DataFrame()
with timer('Read part 1 data'):
    frame1 = pd.read_hdf(FoldPath+'MeanFlow4.h5')
frame2 = pd.read_hdf(FoldPath+'MeanFlow5.h5')
frame3 = pd.read_hdf(FoldPath+'MeanFlow6.h5')
frame4 = pd.read_hdf(FoldPath+'MeanFlow7.h5')
a, b, c, d = 1, 1, 1, 1
SumFrame = frame1*a+frame2*b+frame3*c+frame4*d
MeanFrame = SumFrame/4
MeanFrame.to_hdf(OutFolder + "MeanFlow.h5", 'w', format='fixed')
"""
# %% convert h5 to szplt for spanwise-average data
"""
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/7/"
OutFolder = "/media/weibo/Data3/szplt/"
# dirs = os.scandir(FoldPath)
dirs = sorted(os.listdir(FoldPath))
os.chdir(FoldPath)
time = np.arange(770.0, 879.5 + 0.5, 0.5)
for i, folder in enumerate(dirs):
    outfile = os.path.splitext(folder)[0]
    with timer("Read "+folder+" data"):
        dataframe = pd.read_hdf(folder)
        newframe1 = dataframe.query("x<=0.0 & y>=0.0")
        newframe2 = dataframe.query("x>=0.0")
        p2p.frame2szplt(
            newframe1, OutFolder, outfile + 'A', time=time[i], zonename=1)
        p2p.frame2szplt(
            newframe2, OutFolder, outfile + 'B', time=time[i], zonename=2)
"""
# %% Save boundary layer profile
"""
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/7/"
datafra = pd.read_hdf(FoldPath+'MeanFlow7.h5')
newframe = datafra.groupby(['x', 'y'])
mean = newframe.mean().reset_index()
frame = mean.loc[ mean['x']==-40.0 ]
newfra = frame[['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'Mach', 'T']]
newfra.to_hdf(FoldPath+'Profile1.h5', 'w', format='fixed')
newfra.to_csv(FoldPath+'Profile.dat', header=newfra.columns, 
              index=False, sep='\t', mode='w')
"""
#%%
"""
VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'div', 'vorticity_1',
           'vorticity_2', 'vorticity_3', 'shear', 'Q-criterion',
           'L2-criterion', 'grad(rho)_1', 'grad(rho)_2', 'grad(rho)_3',
           '|grad(rho)|', 'Mach', 'entropy', 'T']
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/7/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
with timer("Read data"):
    DataFrame = p2p.NewReadINCAResults(240, FoldPath+"TP_data_01030465/", VarList,
                                       OutFolder)
#DataFrame.to_hdf(OutFolder+"MeanFlow7.h5", 'w', format='fixed')
"""

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
    'T',
]
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/5/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/51/"
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
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/4/01/"
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

MeanFrame.to_hdf(OutFolder+"MeanFlow43.h5", 'w', format='fixed')
"""
# %% Time-average DataFrame
"""
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/04/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
dirs = os.scandir(FoldPath)
num = np.size(os.listdir(FoldPath))
SumFrame = pd.DataFrame()
for folder in dirs:
    path = FoldPath+folder.name
    with timer("Read "+folder.name+" data"):
        DataFrame = pd.read_hdf(path)
    SumFrame = SumFrame.add(DataFrame, fill_value=0)

MeanFrame = SumFrame/num
MeanFrame.to_hdf(OutFolder + "MeanFlow.h5", 'w', format='fixed')

FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/04/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
with timer("Read dataframe"):
    dataframe = pd.read_hdf(OutFolder+'MeanFlow4.h5')
with timer("Convert .h5 to tecplot .dat"):   
    p2p.frame2tec(dataframe, OutFolder, 'MeanFlow4')
"""

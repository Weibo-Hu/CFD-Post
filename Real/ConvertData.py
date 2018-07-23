"""
Created on Sat Jul 4 13:30:50 2018
    This code for preprocess data for futher use.
@author: Weibo Hu
"""
# %% Load necessary module
import os
from timer import timer
import plt2pandas as p2p

# %% Make spanwise-averaged snapshots
VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'T']
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/5/04/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/5/04/"
NoBlock = 240
dirs1 = os.listdir(FoldPath)
dirs = os.scandir(FoldPath)
for folder in dirs:
    path = FoldPath+folder.name+"/"
    with timer("Read "+folder.name+" data"):
        DataFrame = p2p.NewReadINCAResults(NoBlock, path, VarList,
                                           OutFolder, SpanAve="Yes")

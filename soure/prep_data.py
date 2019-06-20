"""
Created on Sat Jul 4 13:30:50 2018
    This code for preprocess data for futher use.
@author: Weibo Hu
"""
# %% Load necessary module
import os
from timer import timer
import tecplot as tp
import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
from glob import glob

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
    '|gradp|'
]

#VarList = [
#    'x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<p>', '<T>', '<u`u`>',
#    '<u`v`>', '<u`w`>', '<v`v`>', '<v`w`>',
#    '<w`w`>', '<Q-criterion>', '<lambda_2>', '|<gradp>|'
#]
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)'
# equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
FoldPath = "/media/weibo/VID1/BFS_M1.7TS/Slice/1/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/"
OutFolder1 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5/"
OutFolder2 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5B/"
OutFolder3 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5C/"
OutFolder4 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5D/"
OutFolder5 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5E/"
OutFolder6 = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/5F/"
NoBlock = 606
# dirs1 = os.listdir(FoldPath)

dirs = os.scandir(FoldPath)
for folder in dirs:
    path = FoldPath+folder.name+"/"
    with timer("Read " + folder.name + " data"):
        DataFrame, time = \
        p2p.ReadAllINCAResults(5, path, # VarList, SpanAve='Yes',
                               FoldPath2=FoldPath, Equ=equ)
        #df1 = p2p.SaveSlice(DataFrame, time, 2.0, OutFolder1)
        #df2 = p2p.SaveSlice(DataFrame, time, 1.5, OutFolder1)
        #df3 = p2p.SaveSlice(DataFrame, time, 1.0, OutFolder1)
        #df4 = p2p.SaveSlice(DataFrame, time, 0.5, OutFolder1)
        #df5 = p2p.SaveSlice(DataFrame, time, 0.0, OutFolder1)
        #df6 = p2p.SaveSlice(DataFrame, time, -2.0, OutFolder1)

#DataFrame.to_csv(FoldPath + "MeanFlow.dat", sep="\t", index=False,
#                 header=VarList, float_format='%.10e')
# %% Extract Data for 3D DMD
"""

VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    '|gradp|'
]

equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
FoldPath = "/media/weibo/Data2/BFS_M1.7L/TP_data_00895454/"
path2 = "/media/weibo/Data3/BFS_M1.7L_0505/3DSnapshots/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/"

NoBlock = 606
skip = 0
cube = [(-10.0, 30.0), (-3.0, 1.0), (-8, 8)]
dirs = os.listdir(FoldPath)
# FileId = p2p.ExtractZone(FoldPath + dirs[0] + "/", cube, NoBlock, skip=skip)
FileId = p2p.ExtractZone(FoldPath, cube, NoBlock, skip=skip)
FileId.to_csv(FoldPath+str(skip)+'ReadList.dat', index=False, sep='\t')
FileId['name'].to_csv(FoldPath + 'ZoomZone.dat', index=False)

# FileId = pd.read_csv(path2 + str(skip)+ "ReadList.dat", sep='\t')
#
# dirs = os.scandir(FoldPath)
# for folder in dirs:
#     path = FoldPath+folder.name+"/"
#     with timer("Read " + folder.name + " data"):
#         DataFrame, time = \
#         p2p.NewReadINCAResults(NoBlock, path, VarList,
#                                FileName=FileId['name'].tolist(),
#                                SavePath=OutFolder, Equ=equ,
#                                skip=skip)

"""
# %% merge snapshots into a single file
path = '/media/weibo/VID1/BFS_M1.7TS/'
snap = 'TP_2D_Z_03'
FoldPath = path + 'snap/range1/'
OutPath = path + 'Slice/' + snap + '/'
dirs = os.listdir(FoldPath)

num = np.size(dirs)
for i in range(0, num):
    path = FoldPath + dirs[i] + "/"
    file = glob(path + snap + '_*.plt')
    with timer("merge " + str(i) + " file"):
        dataset = tp.data.load_tecplot(file, read_data_option=2)
        SolTime = dataset.solution_times[0]
        tm = '_' + "%08.2f" % SolTime
        tp.data.save_tecplot_plt(OutPath + snap + tm + '.plt', dataset=dataset)
        # tp.data.save_tecplot_szl(path + snap + '.szplt', dataset=dataset)
        

# %% Save time-averaged flow field
"""
VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'div', 'vorticity_1',
           'vorticity_2', 'vorticity_3', 'shear', 'Q-criterion',
           'L2-criterion', 'grad(rho)_1', 'grad(rho)_2', 'grad(rho)_3',
           '|grad(rho)|', 'Mach', 'entropy', 'T', '|gradp|']
FoldPath = "/media/weibo/Data2/BFS_M1.7C_TS1/snapshots/"
OutFolder = "/media/weibo/Data2/BFS_M1.7C_TS1/DataPost/"
dirs = os.scandir(FoldPath)
num = np.size(os.listdir(FoldPath))
# equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)'
for i, folder in enumerate(dirs):
    path = FoldPath+folder.name+"/"
    if i == 0:
        with timer("Read "+folder.name+" data"):
            SumFrame = p2p.ReadINCAResults(422, path, VarList, Equ=equ,
                                           SavePath=OutFolder)[0]
    else:
        with timer("Read "+folder.name+" data"):
            DataFrame = p2p.ReadINCAResults(422, path, VarList, Equ=equ,
                                            SavePath=OutFolder)[0]
        if (np.shape(DataFrame['x']) != np.shape(SumFrame['x'])):
            sys.exit("DataFrame does not match!!!")
        else:
            SumFrame = SumFrame + DataFrame

MeanFrame = SumFrame/num

MeanFrame.to_hdf(OutFolder+"MeanFlow2.h5", 'w', format='fixed')
"""
# %% Time-average DataFrame
"""
FoldPath = "/media/weibo/Data3/BFS_M1.7L_0505/Snapshots/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/MeanFlow/"
dirs = os.scandir(FoldPath)
num = np.size(os.listdir(FoldPath))
SumFrame = pd.DataFrame()
for folder in dirs:
    with timer('Read'+folder.name):
        frame = pd.read_hdf(FoldPath+folder.name)
        SumFrame = SumFrame.append(frame)

grouped = SumFrame.groupby(['x', 'y', 'z'])
TimeAve = grouped.mean().reset_index()
TimeAve.to_hdf(OutFolder + "TimeAve.h5", 'w', format='fixed')

grouped1 = TimeAve.groupby(['x', 'y'])
Meanframe = grouped1.mean().reset_index()
Meanframe.to_hdf(OutFolder + "MeanFlow.h5", 'w', format='fixed')
"""
# %% convert h5 to plt for spanwise-average data
"""
FoldPath = "/media/weibo/Data2/BFS_M1.7C_L1/DataPost/6/"
FoldPath1 = "/media/weibo/Data2/BFS_M1.7C_L1/DataPost/"
OutFolder = "/media/weibo/Data2/BFS_M1.7C_L1/DataPost/"
# dirs = os.scandir(FoldPath)
dirs = sorted(os.listdir(FoldPath))
os.chdir(FoldPath)
time = np.arange(330, 769.50 + 0.5, 0.5)
for i, folder in enumerate(dirs):
    ii = time[i]
    outfile = 'SolTime'+f'{ii:.2f}'
    #outfile = os.path.splitext(folder)[0]
    with timer("Read "+folder+" data"):
        dataframe = pd.read_hdf(folder)
        newframe1 = dataframe.query("x<=0.0 & y>=0.0")
        newframe2 = dataframe.query("x>=0.0")
        p2p.frame2tec(newframe1, FoldPath1, outfile + 'A', time=ii, z=0, zonename=1)
        p2p.frame2tec(newframe2, FoldPath1, outfile + 'B', time=ii, z=0, zonename=2)
    FileName = [FoldPath1+outfile+'A'+'.plt', FoldPath1+outfile+'B'+'.plt']
    dataset = tp.data.load_tecplot(FileName, read_data_option=2)
    # tp.data.operate.execute_equation('{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)')
    tp.data.save_tecplot_plt(OutFolder+outfile+'.plt', dataset=dataset)
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
"""
"""
VarList = [
    'x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<p>', '<T>', '<u`u`>',
    '<u`v`>', '<u`w`>', '<v`v`>', '<v`w`>',
    '<w`w`>', '<Q-criterion>', '<lambda_2>'
]
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2+ddz({p})**2)'
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TP_stat/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
with timer("Read data"):
    DataFrame, soltime = p2p.NewReadINCAResults(240, FoldPath, VarList)
grouped = DataFrame.groupby(['x', 'y'])
df = grouped.mean().reset_index()
df.to_hdf(OutFolder+"SpanAveTP_stat.h5", 'w', format='fixed')
"""

"""
VarList = [
    'x', 'y', 'z', '<u>', '<v>', '<w>', '<rho>', '<p>', '<T>', '<u`u`>',
    '<u`v`>', '<u`w`>', '<v`v`>', '<v`w`>',
    '<w`w`>', '<Q-criterion>', '<lambda_2>', '|<gradp>|'
]
"""
# %% convert h5 to szplt for spanwise-average data
"""
FoldPath = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/plt/8/"
FoldPath1 = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/plt/10/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/plt/10/"
# dirs = os.scandir(FoldPath)
dirs = sorted(os.listdir(FoldPath))
os.chdir(FoldPath)
time = np.arange(880.00, 989.50 + 0.5, 0.5)
for i in range(np.size(time)):
    with timer("Read "+str(time[i])+" data"):
        ii = time[i]
        FileName = FoldPath+'SolTime'+f'{ii:.2f}'+'.plt'
        dataset = tp.data.load_tecplot(FileName, read_data_option=2)
        #tp.data.operate.execute_equation('{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)')
        dataset.zone(0).solution_time = ii
        dataset.zone(0).strand = 1
        dataset.zone(1).solution_time = ii
        dataset.zone(1).strand = 1
        tp.data.save_tecplot_plt(OutFolder+'SolTime'+f'{ii:.2f}'+'.plt', dataset=dataset)
"""

"""
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/plt/8/"
FoldPath1 = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/8/"
OutFolder = "/media/weibo/Data3/BFS_M1.7L_0505/SpanAve/plt/10/"
# dirs = os.scandir(FoldPath)
dirs = sorted(os.listdir(FoldPath))
os.chdir(FoldPath)
#VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'div', 'vorticity_1',
#       'vorticity_2', 'vorticity_3', 'shear', 'Q-criterion', 'L2-criterion',
#       'grad(rho)_1', 'grad(rho)_2', 'grad(rho)_3', '|grad(rho)|', 'Mach',
#       'entropy', 'T', '|gradp|']
VarList = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'vorticity_1', 'vorticity_2',
           'vorticity_3', 'Q-criterion', 'L2-criterion', 'T', '|gradp|']
for i, folder in enumerate(dirs):
    with timer("Read"+folder+" data"):
        outfile = os.path.splitext(folder)[0]
        Data = p2p.NewReadINCAResults(2, FoldPath+folder, VarList, FoldPath1,
                                      OutFile='MeanFlow1')
        Data.to_hdf(FoldPath1 + outfile + ".h5", 'w', format='fixed')
"""

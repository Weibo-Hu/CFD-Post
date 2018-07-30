# -*- coding: utf-8 -*-
"""
Created on Sat Jun 9 10:24:50 2018
    This code for reading binary data from tecplot (.plt) and 
    Convert data to pandas dataframe
@author: Weibo Hu
"""
import tecplot as tp
import pandas as pd
import os
import numpy as np
from DataPost import DataPost
from timer import timer

#   Show Progress of code loop
def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('%s %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def ReadPlt(FoldPath, VarList):
    FileList = os.listdir(FoldPath)
    #clear dataset first
    for j in range(np.size(FileList)):
        FileName = FoldPath + FileList[j]
        dataset  = tp.data.load_tecplot(FileName)
        #namelist = dataset.VariablesNamedTuple
        zone = dataset.zone
        #zones = dataset.zones()
        zonename = zone(j).name
        #print(zonename)
        for i in range(np.size(VarList)):
            var  = dataset.variable(VarList[i])
            if i == 0:
                VarCol = var.values(zonename).as_numpy_array()
                #print(np.size(VarCol))
            else:
                Var_index = var.values(zonename).as_numpy_array()
                VarCol = np.column_stack((VarCol, Var_index))
        if j == 0:
            SolTime = dataset.solution_times
            #print(SolTime)
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
        del dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    return(df)


def ReadINCAResults(BlockNO, FoldPath, VarList, FoldPath2, \
                    SpanAve=None, OutFile=None):
    for j in range(BlockNO):
        #progress(j, BlockNO, 'Read *.plt:')
        FileName = FoldPath + "TP_dat_"+str(j+1).zfill(6)+".plt"
        dataset = tp.data.load_tecplot(FileName)
        zone = dataset.zone
        zonename = zone(j).name
        for i in range(np.size(VarList)):
            var  = dataset.variable(VarList[i])
            if i == 0:
                VarCol = var.values(zonename).as_numpy_array()
            else:
                Var_index = var.values(zonename).as_numpy_array()
                VarCol = np.column_stack((VarCol, Var_index))
        if j == 0:
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
        SolTime = dataset.solution_times[-1]
        del FileName, dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    #print(SolTime)
    #df.to_csv(OutFile+".dat", index=False, sep = '\t')
    if SpanAve is not None:
        grouped = df.groupby(['x', 'y'])
        df      = grouped.mean().reset_index()
#        df = df.loc[df['z'] == 0.0].reset_index(drop=True)
    if OutFile is None:
        df.to_hdf(FoldPath2+"SolTime"+str(round(SolTime,2))+".h5", \
                  'w', format= 'fixed')
    #else:
    #    df.to_hdf(FoldPath2 + OutFile + ".h5", 'w', format='fixed')
    return(df)


def NewReadINCAResults(BlockNO, FoldPath, VarList, FoldPath2, \
                    SpanAve=None, OutFile=None):
    os.chdir(FoldPath)
    FileName = sorted(os.listdir(FoldPath))
    dataset = tp.data.load_tecplot(FileName, read_data_option=2)
    SolTime = dataset.solution_times[0]
    if (np.size(FileName) != BlockNO):
        sys.exit("You're missing some blocks!!!")
    for j in range(BlockNO):
        zone = dataset.zone
        zonename = zone(j).name
        for i in range(np.size(VarList)):
            var = dataset.variable(VarList[i])
            if i == 0:
                VarCol = var.values(zonename).as_numpy_array()
            else:
                Var_index = var.values(zonename).as_numpy_array()
                VarCol = np.column_stack((VarCol, Var_index))
        if j == 0:
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
    del FileName, dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    #print(SolTime)
    #df.to_csv(OutFile+".dat", index=False, sep = '\t')
    if SpanAve is not None:
        grouped = df.groupby(['x', 'y'])
        df = grouped.mean().reset_index()

#        df = df.loc[df['z'] == 0.0].reset_index(drop=True)
    if OutFile is None:
        df.to_hdf(FoldPath2+"SolTime"+"%0.2f"%SolTime+".h5", \
                  'w', format= 'fixed')
    #else:
    #    df.to_hdf(FoldPath2 + OutFile + ".h5", 'w', format='fixed')
    return (df)


def ReadAllINCAResults(BlockNO, FoldPath, FoldPath2, \
                    SpanAve=None, OutFile=None):
    os.chdir(FoldPath)
    FileName = os.listdir(FoldPath)
    dataset = tp.data.load_tecplot(FileName, read_data_option=2)
    VarList = [v.name for v in dataset.variables()]
    zone = dataset.zone
    if (np.size(FileName) != BlockNO):
        sys.exit("You're missing some blocks!!!")
    for j in range(BlockNO):
        zonename = zone(j).name
        for i in range(np.size(VarList)):
            var = dataset.variable(VarList[i])
            if i == 0:
                VarCol = var.values(zonename).as_numpy_array()
            else:
                Var_index = var.values(zonename).as_numpy_array()
                VarCol = np.column_stack((VarCol, Var_index))
        if j == 0:
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
    del FileName, dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    if SpanAve is not None:
        grouped = df.groupby(['x', 'y'])
        df = grouped.mean().reset_index()
#        df = df.loc[df['z'] == 0.0].reset_index(drop=True)
    if OutFile is not None:
        df.to_hdf(FoldPath2 + OutFile + ".h5", 'w', format='fixed')
    return(df)


def frame2tec(dataframe, SaveFolder, FileName, float_format='%.8f'):
    if not os.path.exists(SaveFolder):
        raise IOError('ERROR: directory does not exist: %s' % SaveFolder)
    SavePath = os.path.join(SaveFolder, FileName)
    header = "VARIABLES="
    zone_name = "Zone_0001"
    zone = 'ZONE T= "{}" \n\n'.format(zone_name)
    I = np.size(dataframe.x.unique())
    J = np.size(dataframe.y.unique())
    K = np.size(dataframe.z.unique())
    for i in range(len(dataframe.columns)):
        header = '{} "{}"'.format(header, dataframe.columns[i])
    with open(SavePath + '.dat', 'w') as f:
        f.write(header+'\n')
        f.write(zone)
        f.write('I = {}, J = {}, K = {} \n'.format(I, J, K))
        dataframe.to_csv(f, index=False, header=False,
                         float_format=float_format)



def tec2plt(Folder, InFile, OutFile):
    dataset = tp.data.load_tecplot(Folder + InFile + '.dat')
    tecplot.data.save_tecplot_plt(OutFile + '.plt')


def tec2szplt(Folder, InFile, OutFile):
    dataset = tp.data.load_tecplot(Folder + InFile + '.dat')
    tecplot.data.save_tecplot_szl(OutFile + '.szplt')


def frame2plt(dataframe, SaveFolder, OutFile, float_format='%.8f'):
    frame2tec(dataframe, SaveFolder, OutFile)
    tec2plt(SaveFolder, OutFile, OutFile)


def frame2szplt(dataframe, SaveFolder, OutFile, float_format='%.8f'):
    frame2tec(dataframe, SaveFolder, OutFile)
    tec2szplt(SaveFolder, OutFile, OutFile)


# Obtain Spanwise Average Value of Data
def SpanAve(DataFrame, OutputFile = None):
    grouped = DataFrame.groupby(['x', 'y'])
    DataFrame = grouped.mean().reset_index()
    if OutputFile is not None:
        outfile  = open(OutputFile, 'x')
        DataFrame.to_csv(outfile, index=False, sep = '\t')
        outfile.close()

#VarList  = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'T']
#FoldPath = "/media/weibo/Data1/BFS_M1.7L_0419/4/"
#OutFolder = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/"
#dirs = os.listdir(FoldPath)
#num = np.size(dirs)
#for ii in range(num):
#    progress(ii, num, ' ')
#    path  = FoldPath+dirs[ii]+"/"
#    with timer("Read .plt data"):
#        DataFrame = ReadINCAResults(214, path, VarList, OutFolder, SpanAve="Yes")

#%% Read plt data from INCA
#FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/TP_stat/"
#OutFile  = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/"
#VarList  = ['x', 'y', '<u>', '<v>', '<w>', '<rho>', '<p>', '<T>', '<u`u`>', \
#            '<u`v`>', '<u`w`>', '<v`v`>', '<v`w`>', '<w`w`>', '<Q-criterion>']
#with timer("Read Meanflow data"):
#    stat = ReadPlt(FoldPath, VarList)
#stat.to_hdf(OutFile+"Meanflow" + ".h5", 'w', format='fixed')
#stat.to_csv(OutFile+"Meanflow.dat", index=False, sep = '\t')

#%% Load all plt files at once
#VarList  = ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'Q-criterion', 'L2-criterion', Mach', 'T']
#FoldPath = "/media/weibo/Data1/BFS_M1.7L_0419/1/00/"
#OutFolder = "/media/weibo/Data1/BFS_M1.7L_0419/TimeAve/"
#dirs = os.listdir(FoldPath)
#num = np.size(dirs)
#workdir = os.chdir(FoldPath+dirs[0])
#FileName = os.listdir(workdir)
#dataset = tp.data.load_tecplot(FileName)

#%% Save time-averaged flow field
#VarList = [
#    'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'Q-criterion', 'L2-criterion',
#    'Mach', 'T'
#]
#FoldPath = "/media/weibo/Data1/BFS_M1.7L_0505/3/"
#OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
#dirs = os.listdir(FoldPath)
#num = np.size(dirs)
#MeanFrame = ReadAllINCAResults(214, FoldPath+dirs[0]+"/", OutFolder)
#for ii in range(num-1):
#    progress(ii, num, ' ')
#    path  = FoldPath+dirs[ii+1]+"/"
#    with timer("Read .plt data"):
#        DataFrame = ReadINCAResults(214, path, VarList, OutFolder)
#        MeanFrame = pd.concat((MeanFrame, DataFrame))
#        MeanFrame = MeanFrame.groupby(level=0).mean()
#        #SumFrame  = SumFrame.add(DataFrame, fill_value=0)
#MeanFrame.to_hdf(OutFolder+"MeanFlow.h5", 'w', format='fixed')

#%% Save timeseries data
#InFolder  = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/1/"
#OutFolder = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/"
#timepoints = np.linspace(200, 203, 4)
#dirs = os.listdir(InFolder)
#Snapshots = pd.read_hdf(InFolder+dirs[0])
#Snapshots['time'] = timepoints[0]
#for jj in range(np.size(dirs)-1):
#    DataFrame = pd.read_hdf(InFolder+dirs[jj+1])
#    DataFrame['time'] = timepoints[jj+1]
#    #Snapshots.append(DataFrame)
#    Snapshots = pd.concat([Snapshots, DataFrame])
#    del DataFrame
#
#Snapshots.to_hdf(OutFolder+"TimeSeries.h5", 'w', format='fixed')
#stat = pd.read_hdf(OutFolder+"TimeSeries.h5")

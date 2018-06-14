# -*- coding: utf-8 -*-
"""
Created on Sat Jun 9 10:24:50 2018
    This code for reading binary data from tecplot (.plt) and 
    Convert data to pandas dataframe
@author: Weibo Hu
"""
import tecplot as tp
import pandas as pd
import sys, time, os
import numpy as np
from DataPost import DataPost

#   Show Progress of code loop
def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('%s %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def ReadPlt(FoldPath, InFile, VarList, OutFile=None):
    #datafile = os.path.join(FilePath, FileName)
    #clear dataset first
    start_time = time.clock()
    for j in range(np.size(FileList)):
        dataset = tp.data.load_tecplot(FileList[j])
        #namelist = dataset.VariablesNamedTuple
        zone = dataset.zone
        #zones = dataset.zones()
        zonename = zone(j).name
        #print(zonename)
        for i in range(np.size(VarList)):
            var  = dataset.variable(VarList[i])
            if i == 0:
                VarCol = var.values(zonename).as_numpy_array()
                print(np.size(VarCol))
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
    if OutFile is None:
        df.to_csv(FoldPath+"SolTime"+str(round(SolTime[0],1))+".dat", \
                  index=False, sep = '\t')
    else:
        df.to_csv(FoldPath+OutFile+".dat", index=False, sep = '\t')
    #df.to_hdf(OutFile+".h5", 'w', format= 'fixed')
    #hdf1 = pd.read_hdf(OutFile+'.h5')
    #dat1 = pd.read_csv(OutFile+'.dat', sep='\t',skiprows=1, skipinitialspace=True)
    print("The cost time of reading plt data", time.clock()-start_time, '\n')
    return(df)


def ReadINCAResults(BlockNO, FoldPath, VarList, FoldPath2, \
                    SpanAve=None, OutFile=None):
    start_time = time.clock()
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
    if OutFile is None:
        df.to_hdf(FoldPath2+"SolTime"+str(round(SolTime,1))+".h5", \
                  'w', format= 'fixed')
    else:
        df.to_hdf(FoldPath2 + OutFile + ".h5", 'w', format='fixed')
    print("The cost time of reading plt data ", time.clock()-start_time, '\n')
    return(df)

# Obtain Spanwise Average Value of Data
def SpanAve(DataFrame, OutputFile = None):
    start_time = time.clock()
    grouped = DataFrame.groupby(['x', 'y'])
    DataFrame = grouped.mean().reset_index()
    if OutputFile is not None:
        outfile  = open(OutputFile, 'x')
        DataFrame.to_csv(outfile, index=False, sep = '\t')
        outfile.close()
    print("The spanwise-averaged time is ", time.clock()-start_time, '\n')

VarList  = ['x', 'y', 'z', 'u', 'v', 'w', 'p', 'T']
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0419/4/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0419/SpanAve/"
dirs = os.listdir(FoldPath)
num = np.size(dirs)
for ii in range(num):
    progress(ii, num, 'Completed folder:')
    path  = FoldPath+dirs[ii]+"/"
    DataFrame = ReadINCAResults(214, path, VarList, OutFolder, SpanAve="Yes")

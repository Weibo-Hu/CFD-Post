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
import time

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
        print(zonename)
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
            print(SolTime)
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
    print("The cost time of reading plt data ", time.clock()-start_time)
    return(df)


def ReadINCAResults(BlockNO, FoldPath, VarList, OutFile=None):
    start_time = time.clock()
    for j in range(BlockNO):
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
            SolTime = dataset.solution_times
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
        del FileName, dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    #df.to_csv(OutFile+".dat", index=False, sep = '\t')
    if OutFile is None:
        df.to_hdf(FoldPath+"SolTime"+str(round(SolTime[0],1))+".h5", 'w', format= 'fixed')
    else:
        df.to_hdf(OutFile +".h5", 'w', format= 'fixed')
    print("The cost time of reading plt data ", time.clock()-start_time)
    return(df)

FileList = ["TP_dat_000005.plt", "TP_dat_000001.plt", "TP_dat_000078.plt"]
VarList  = ['x', 'y', 'z']
FoldPath = "/media/weibo/Data1/BFS_M1.7L_0419/2/TP_data_00336480/"
OutFile = "1"
path  = "./"
#ReadPlt(path, FileList, ['x', 'y', 'z'])
ReadINCAResults(214, FoldPath, VarList)


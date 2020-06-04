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
import sys
import warnings
import numpy as np
from scipy.interpolate import griddata
from timer import timer
from time import time
import logging as log
from glob import glob

log.basicConfig(level=log.INFO)

#   Show Progress of code loop
def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('%s %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def ReadPlt(FoldPath, VarList=None):
    FileList = os.listdir(FoldPath)
    #clear dataset first
    for j in range(np.size(FileList)):
        FileName = FoldPath + FileList[j]
        dataset  = tp.data.load_tecplot(FileName)
        #namelist = dataset.VariablesNamedTuple
        zone = dataset.zone
        #zones = dataset.zones()
        zonename = zone(j).name
        if VarList is None:
            VarList = [v.name for v in dataset.variables()]
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
            if (np.size(dataset.solution_times) == 0):
                SolTime = 0.0
            else:
                SolTime = dataset.solution_times[0]
            #print(SolTime)
            ZoneRow = VarCol
        else:
            ZoneRow = np.row_stack((ZoneRow, VarCol))
        del dataset, zone, zonename, var
    df = pd.DataFrame(data=ZoneRow, columns=VarList)
    return(df)


def ReadINCAResults(FoldPath, VarList, SubZone=None, FileName=None, Equ=None,
                    SpanAve=None, SavePath=None, OutFile=None, skip=0, opt=2):
    if FileName is None:
        files = sorted(os.listdir(FoldPath))
        FileName = [os.path.join(FoldPath, name) for name in files]
    if(isinstance(FileName, list)):
        szplt = FileName[0].find('szplt')
    else:
        szplt = FileName.find('szplt')
    if(szplt != -1):
        dataset = tp.data.load_tecplot_szl(FileName, read_data_option=2)
    else:
        dataset = tp.data.load_tecplot(FileName, read_data_option=2)
    if Equ is not None:
        for i in range(np.size(Equ)):
            tp.data.operate.execute_equation(Equ[i])
    if (np.size(dataset.solution_times) == 0):
        SolTime = 0.0
    else:
        SolTime = dataset.solution_times[0]
    skip = skip + 1
    # num_zones = dataset.num_zones
    df = pd.DataFrame(columns=VarList)
    for zone in dataset.zones('*'):
        xvar = zone.values('x').as_numpy_array()
        yvar = zone.values('y').as_numpy_array()
        zvar = zone.values('z').as_numpy_array()
        nx = int(np.size(np.unique(xvar)))
        ny = int(np.size(np.unique(yvar)))
        nz = int(np.size(np.unique(zvar)))
        x1 = np.min(xvar)
        x2 = np.max(xvar)
        y1 = np.min(yvar)
        y2 = np.max(yvar)
        z1 = np.min(zvar)
        z2 = np.max(zvar)
        if SubZone is not None:
            withinzone = ((SubZone[0][0] < x2 and x1 < SubZone[0][1])
                     and (SubZone[1][0] < y2 and y1 < SubZone[1][1])
                     and (SubZone[2][0] < z2 and z1 < SubZone[2][1]))

        if(SubZone is None or withinzone==True):
            for i in range(np.size(VarList)):
                varval = zone.values(VarList[i]).as_numpy_array()
                # this method does much repeated work,
                # try to find index to filter variables
                if skip != 1:
                    NewCol = varval.reshape((nx, ny, nz), order='F')
                    if nx % skip == 1:
                        NewCol = NewCol[0::skip, :, :]
                    else:
                        print("No skip in x direction")
                    if ny % skip == 1:
                        NewCol = NewCol[:, 0::skip, :]
                    else:
                        print("No skip in y direction")
                    if nz % skip == 1:
                        NewCol = NewCol[:, :, 0::skip]
                    else:
                        print("No skip in z direction")
                    varval = NewCol.ravel(order='F')

                if i == 0:  # first column
                    VarCol = varval
                else:  # other columns
                    Var_index = varval
                    VarCol = np.column_stack((VarCol, Var_index))
            df1 = pd.DataFrame(data=VarCol, columns=VarList)
            df = df.append(df1, ignore_index=True)
    del dataset, varval
    # df = df.drop_duplicates(keep='last')  # if on,spanwise-average may wrong
    if SpanAve == True:
        grouped = df.groupby(['x', 'y'])
        df = grouped.mean().reset_index()
    if SpanAve == None:
        grouped = df.groupby(['x', 'y', 'z'])
        df = grouped.mean().reset_index()
#        df = df.loc[df['z'] == 0.0].reset_index(drop=True)
    if opt == 2:
        df['x'] = df['x'].astype(float)
        df['y'] = df['y'].astype(float)
        df['z'] = df['z'].astype(float)
        if SubZone is not None:
            df = df.query("x>={0} & x<={1} & y<={2}".format(
                          SubZone[0][0], SubZone[0][1], SubZone[1][1]))
    if SavePath is not None and OutFile is not None:
        st = "%08.2f" % SolTime
        df.to_hdf(SavePath+OutFile+'_'+st+".h5", 'w', format='fixed')
    return(df, SolTime)


# %% save zone information of every tecplot file
def save_zone_info(path, filename=None):
    init = os.scandir(path)
    cols = ['x1', 'x2', 'y1', 'y2', 'z1',
            'z2', 'nx', 'ny', 'nz']
    name = np.empty(shape=[0, 1])
    boundary = np.empty(shape=[0, 9])
    for folder in init:
        file = path + folder.name
        dataset = tp.data.load_tecplot(file, read_data_option=2)
        zone = dataset.zone
        zonename = zone(0).name
        var_x = dataset.variable('x').values(zonename).as_numpy_array()
        var_y = dataset.variable('y').values(zonename).as_numpy_array()
        var_z = dataset.variable('z').values(zonename).as_numpy_array()
        nx = int(np.size(np.unique(var_x)))
        ny = int(np.size(np.unique(var_y)))
        nz = int(np.size(np.unique(var_z)))
        nxyz = np.size(var_x)
        if nxyz != nx * ny * nz:
            sys.exit("The shape of data does not match!!!")
        x1 = np.min(var_x)
        x2 = np.max(var_x)
        y1 = np.min(var_y)
        y2 = np.max(var_y)
        z1 = np.min(var_z)
        z2 = np.max(var_z)

        name = np.append(name, folder.name)
        information = [x1, x2, y1, y2, z1, z2, nx, ny, nz]
        boundary = np.vstack((boundary, information))
    name = name.reshape(-1, 1)
    df = pd.DataFrame(data=boundary, columns=cols)
    df['name'] = name
    if filename is not None:
        df.to_csv(filename, index=False, sep=' ')
    return (df)

# %% save index information for convert .h5 to plt
def save_tec_index(df_data, df_zone_info, filename=None):
    dat = np.empty(shape=[0, 2])
    for jj in range(np.shape(df_zone_info)[0]):
        file = df_zone_info.iloc[jj]
        # extract zone according to coordinates
        df_id = df_data.query("x>={0} & x<={1} & y>={2} & y<={3}".format(
                              file['x1'], file['x2'], file['y1'], file['y2']))
        ind = df_id.index.values
        file_id = np.ones(np.size(ind)) * (jj + 1)
        temp = np.vstack((file_id, ind))
        dat = np.vstack( (dat, temp.transpose()) )
    df = pd.DataFrame(data=dat, columns=['file', 'ind'])
    if filename is not None:
        df.to_csv(filename, index=False, sep=' ')
        # df.to_hdf(filename, 'w', format='fixed')
    return (df)


def extract_zone(path, cube, skip=0, filename=None):
    # cube = [(-5.0, 25.0), (-3.0, 5.0), (-2.5, 2.5)]
    init = os.scandir(path)
    cols = ['x1', 'x2', 'y1', 'y2', 'z1',
            'z2', 'nx', 'ny', 'nz']
    name = np.empty(shape=[0, 1])
    boundary = np.empty(shape=[0, 9])
    skip = skip + 1
    for folder in init:
        file = path + folder.name
        dataset = tp.data.load_tecplot(file, read_data_option=2)
        zone = dataset.zone
        zonename = zone(0).name
        var_x = dataset.variable('x').values(zonename).as_numpy_array()
        var_y = dataset.variable('y').values(zonename).as_numpy_array()
        var_z = dataset.variable('z').values(zonename).as_numpy_array()
        nx = int(np.size(np.unique(var_x)))
        ny = int(np.size(np.unique(var_y)))
        nz = int(np.size(np.unique(var_z)))
        nxyz = np.size(var_x)
        if nxyz != nx * ny * nz:
            sys.exit("The shape of data does not match!!!")
        x1 = np.min(var_x)
        x2 = np.max(var_x)
        y1 = np.min(var_y)
        y2 = np.max(var_y)
        z1 = np.min(var_z)
        z2 = np.max(var_z)
        # FileID.loc[jj, 'id1'] = ind[0]
        # FileID.loc[jj, 'id2'] = ind[-1]
        if (cube[0][0] < x2 and x1 < cube[0][1]) \
           and (cube[1][0] < y2 and y1 < cube[1][1]) \
           and (cube[2][0] < z2 and z1 < cube[2][1]):
            # id1 = int(id2 + 1)
            # id2 = int(id1 + nx * ny * nz - 1)
            # print(folder.name)
            if skip != 1:
                if nx % skip == 1:
                    nx = (nx + 1) // skip
                else:
                    print("No skip in x direction")
                if ny % skip == 1:
                    ny = (ny + 1) // skip
                else:
                    print("No skip in y direction")
                if nz % skip == 1:
                    nz = (nz + 1) // skip
                else:
                    print("No skip in z direction")
            name = np.append(name, folder.name)
            information = [x1, x2, y1, y2, z1, z2, nx, ny, nz]
            # print(information)
            boundary = np.vstack((boundary, information))
    name = name.reshape(-1, 1)
    df = pd.DataFrame(data=boundary, columns=cols)
    df['name'] = name
    df = df.sort_values(by=['name']).reset_index(drop=True)
    ind1 = np.empty(shape=[0, 1])
    ind2 = np.empty(shape=[0, 1])
    id2 = -1
    for j in range(np.shape(df)[0]):
        id1 = id2 + 1
        id2 = id1 + df.iloc[j]['nx'] * df.iloc[j]['ny'] * df.iloc[j]['nz'] - 1
        ind1 = np.append(ind1, id1)
        ind2 = np.append(ind2, id2)
    df['id1'] = ind1
    df['id2'] = ind2
    if filename is not None:
        df.to_csv(filename, index=False, sep=' ')
    return (df)


def GirdIndex(FileID, xarr, yarr, zarr):
    FileID['nx'] = 0
    FileID['ny'] = 0
    FileID['nz'] = 0
    ind_arr = []
    xyz = pd.DataFrame(np.hstack((xarr, yarr, zarr)), columns=['x', 'y', 'z'])
    for jj in range(FileID.shape[0]):
        file = FileID.iloc[jj]
        # extract zone according to coordinates
        df_id = xyz.query("x>{0} & x<{1} & y>{2} & y<{3}".format(
                          file['x1'], file['x2'], file['y1'], file['y2']))
        # remove duplicate coordinates due to interface (twice at the interface)
        # df_id = df.drop_duplicates()
        ind = df_id.index.values
        ind_arr.append(ind)
        nx = np.size(np.unique(df_id['x'][ind]))
        ny = np.size(np.unique(df_id['y'][ind]))
        nz = np.size(np.unique(df_id['z'][ind]))
        # FileID.loc[jj, 'id1'] = ind[0]
        # FileID.loc[jj, 'id2'] = ind[-1]
        FileID.loc[jj, 'nx'] = nx
        FileID.loc[jj, 'ny'] = ny
        FileID.loc[jj, 'nz'] = nz
    FileID['ind'] = ind_arr
    return (FileID)

def SaveSlice(df, SolTime, SpanAve, SavePath):
    if SpanAve == float('Inf'):  # span-averaged
        grouped = df.groupby(['x', 'y'])
        df = grouped.mean().reset_index()
    else:  # extract a slice at z=SpanAve
        df = df.loc[df['z'] == SpanAve].reset_index(drop=True)
    df.to_hdf(SavePath+"Slice"+"%0.1f"%SpanAve+"SolTime"+"%08.2f"%SolTime+".h5",
              'w', format='fixed')
    return df


def ReadAllINCAResults(FoldPath, SavePath=None, Equ=None,
                       FileName=None, SpanAve=None, OutFile=None):
    if FileName is None:
        # files = os.listdir(FoldPath)
        # FileName = [os.path.join(FoldPath, name) for name in files]
        FileName = glob(FoldPath + '*plt')
    if(isinstance(FileName, list)):
        szplt = FileName[0].find('szplt')
    else:
        szplt = FileName.find('szplt')
    if(szplt != -1):
        dataset = tp.data.load_tecplot_szl(FileName, read_data_option=2)
    else:
        dataset = tp.data.load_tecplot(FileName, read_data_option=2)
    if Equ is not None:
        for i in range(np.size(Equ)):
            tp.data.operate.execute_equation(Equ[i])
    VarList = [v.name for v in dataset.variables()]
    df = pd.DataFrame(columns=VarList)
    if (np.size(dataset.solution_times) == 0):
        SolTime = 0.0
    else:
        SolTime = dataset.solution_times[0]
    for zone in dataset.zones('*'):
        for i in range(np.size(VarList)):
            if i == 0:
                VarCol = zone.values(VarList[i]).as_numpy_array()
            else:
                Var_index = zone.values(VarList[i]).as_numpy_array()
                VarCol = np.column_stack((VarCol, Var_index))
        df1 = pd.DataFrame(data=VarCol, columns=VarList)
        df = df.append(df1, ignore_index=True)

    del FileName, dataset, zone
    # df = df.drop_duplicates(keep='last')
    if SpanAve is not None:
        grouped = df.groupby(['x', 'y'])
        df = grouped.mean().reset_index()
        # df = df.loc[df['z'] == 0.0].reset_index(drop=True)
    if SavePath is not None and OutFile is not None:
        st = "%08.2f" % SolTime
        df.to_hdf(SavePath+OutFile+'_'+st+".h5", 'w', format='fixed')
    return(df)


def frame2tec(dataframe,
              SaveFolder,
              FileName,
              z=None,
              zname=None,
              stime=None,
              float_format='%.8f'):
    if not os.path.exists(SaveFolder):
        raise IOError('ERROR: directory does not exist: %s' % SaveFolder)
    SavePath = os.path.join(SaveFolder, FileName)
    dataframe = dataframe.sort_values(by=['z', 'y', 'x'])
    header = "VARIABLES="
    x = dataframe.x.unique()
    y = dataframe.y.unique()
    if z is None:
        z = dataframe.z.unique()
    I = np.size(x)
    J = np.size(y)
    K = np.size(z)
    if(isinstance(zname, int)):
        zone_name = 'B' + '{:010}'.format(zname)
    else:
        zone_name = 'B' + '{:010}'.format(1)
    zone = 'ZONE T= "{}" \n'.format(zone_name)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    new = np.zeros((I*J*K, np.size(dataframe.columns)))
    if K == 1:
        for i in range(len(dataframe.columns)):
            header = '{} "{}"'.format(header, dataframe.columns[i])
            var = griddata((dataframe.x, dataframe.y),
                           dataframe.values[:, i], (xx, yy), fill_value=0.0)
            new[:, i] = var.flatten('F')
    else:
        temp = np.zeros((I*J, K))
        for i in range(len(dataframe.columns)):
            header = '{} "{}"'.format(header, dataframe.columns[i])
            for j in range(K):
                newframe = dataframe.loc[dataframe['z']==z[j]]
                var = griddata((newframe.x, newframe.y),
                               newframe.values[:, i], (xx, yy))
                temp[:, j] = var.flatten('F')
            new[:, i] = temp.flatten('F')
    new[np.isinf(new)] = 0.0
    new[np.isnan(new)] = 0.0
    if np.isnan(new).any() == True:
        raise ValueError(
                'ERROR: dataframe contains NON value due to geometry',
                'discontinuity, like a step exist in the domain!!!')
    with open(SavePath + '.dat', 'w') as f:
        f.write(header+'\n')
        f.write(zone)
        if stime is not None:
            time = np.float64(stime)
            f.write(' StrandID=1, SolutionTime={}\n\n'.format(time))
        else:
            f.write('\n')
        f.write('I = {}, J = {}, K = {}\n'.format(I, J, K))
        newframe = pd.DataFrame(new, columns=dataframe.columns)
        newframe.to_csv(f, sep='\t', index=False, header=False,
                        float_format=float_format)


def frame2tec3d(dataframe,
                path,
                filename,
                zname=None,
                stime=None,
                float_format='%9.8e'):
    if not os.path.exists(path):
        raise IOError('ERROR: directory does not exist: %s' % path)
    SavePath = os.path.join(path, filename)
    In = np.size(np.unique(dataframe['x']))
    Jn = np.size(np.unique(dataframe['y']))
    Kn = np.size(np.unique(dataframe['z']))

    header = "VARIABLES="
    for i in range(len(dataframe.columns)):
        header = '{} "{}"'.format(header, dataframe.columns[i])
    if(isinstance(zname, int)):
        zonename = 'B' + '{:010}'.format(zname)
    else:
        zonename = 'B' + '{:010}'.format(1)
    zone = 'ZONE T= "{}" \n'.format(zonename)

    with timer("save " + filename + " tecplot .dat"):
        with open(SavePath + '.dat', 'w') as f:
            if stime is not None:
                stime = np.float64(stime)
                timeid = 'StrandID={0}, SolutionTime={1}\n'.format(i+1, stime)
            else:
                timeid = 'StrandID={}\n'.format(i + 1)
            xyz = 'I = {}, J = {}, K = {}\n'.format(In, Jn, Kn)
            f.write(header+'\n')
            f.write(zone)
            f.write(timeid)
            f.write(xyz)
            data = dataframe.sort_values(by=['z', 'y', 'x'])
            data.to_csv(f, sep='\t', index=False, header=False,
                        float_format=float_format)

"""
def zone2tec(path, filename, df, zonename, time=None):
    header = "VARIABLES = "
    zone = 'ZONE T = "{}" \n'.format(zonename)
    for i in range(len(df.columns)):
        header = '{} "{}"'.format(header, df.columns[i])
    with open(path + filename + '.dat', 'w') as f:
        f.write(header+'\n')
        f.write(zone)
        if time is not None:
            time = np.float64(time)
            f.write(' StrandID=1, SolutionTime = {}\n'.format(time))
        else:
            f.write('\n')
        nx = np.size(np.unique(df['x']))
        ny = np.size(np.unique(df['y']))
        nz = np.size(np.unique(df['z']))
        f.write(' I = {}, J = {}, K = {}\n'.format(
                nx, ny, nz))
        df = df.sort_values(by=['z', 'y', 'x'])
        df.to_csv(f, sep=' ', index=False, header=False,
                  float_format='%9.8e')
"""

def mul_zone2tec(path, filename, zoneinfo, df, zoneid=None, stime=None):
    header = "VARIABLES = "
    for j in range(len(df.columns)):
        header = '{} "{}"'.format(header, df.columns[j])
    header = header + '\n'  # write header of the file
    time1 = time()
    for i in range(np.shape(zoneinfo)[0]):
        with open(path + filename + str(i) + '.dat', 'w') as f:
            f.write(header)
            zonename = 'B' + '{:010}'.format(i+1)
            file = zoneinfo.iloc[i]
            zone = 'ZONE T = "{}" \n'.format(zonename)
            if stime is not None:
                stime = np.float64(stime)
                timeid = 'StrandID={0}, SolutionTime={1}\n'.format(i+1, stime)
            else:
                timeid = 'StrandID={}\n'.format(i + 1)
            xyz = 'I = {}, J = {}, K = {}\n'.format(file['nx'],
                                                    file['ny'],
                                                    file['nz'])
            f.write(zone)
            f.write(timeid)
            f.write(xyz)
            if zoneid is None:
                data = df.query(
                    "x>={0} & x<={1} & y>={2} & y<={3}".format(file['x1'],
                                                               file['x2'],
                                                               file['y1'],
                                                               file['y2'])
                )
            else:
                grouped = zoneid.groupby(['file'])
                # ng = grouped.ngroups
                ind = grouped.get_group(i + 1)['ind']
                ind = ind.astype('int')
                data = df.iloc[ind]
            # data = df.iloc[ind1: ind2 + 1]
            data = data.sort_values(by=['z', 'y', 'x'])
            data.to_csv(f, sep='\t', index=False,
                        header=False, float_format='%.8f')
        print("save data " + zonename + " took", time()-time1, "s")


def mul_zone2tec_plt(path, filename, FileId, df, time=None, option=1):
    if option == 1:
        tp.session.connect()
        tp.new_layout()
#        page = tp.active_page()
#        page.name = 'page1'
#        frame = page.active_frame()
#        frame.name = 'frame1'
#        dataset = frame.create_dataset('data1')
        # add variable name
#        for j in range(np.shape(df)[1]):
#            var = df.columns[j]
#            dataset.add_variable(var)
        # link data
        dataset = tp.active_frame().create_dataset('data1', df.columns)
        with tp.session.suspend():
        # with timer("save data as tecplot .plt"):
            for i in range(np.shape(FileId)[0]):
                file = FileId.iloc[i]
                ind1 = int(file['id1'])
                ind2 = int(file['id2'])
                nx = int(file['nx'])
                ny = int(file['ny'])
                nz = int(file['nz'])
                zonename = 'B' + '{:010}'.format(i)
                # print('creating tecplot zone: '+zonename)
                zone = dataset.add_ordered_zone(zonename, (nx, ny, nz))
                # zone = dataset.add_zone('Ordered', zonename, (nx, ny, nz),
                #                         solution_time=time, strand_id=1)
                if time is not None:
                    zone.strand = 1
                    zone.solution_time = np.float64(time)
                data = df.iloc[ind1: ind2 + 1]
                data = data.sort_values(by=['z', 'y', 'x'])
                for j in range(np.shape(data)[1]):
                    var = data.columns[j]
                    zone.values(var)[:] = data[var][:]
        tp.data.save_tecplot_plt(path + filename + '.plt', dataset=dataset)
    else:
        dataset = tp.data.load_tecplot(
            path + filename + '.dat', read_data_option=2)
        tp.data.save_tecplot_plt(path + filename + '.plt', dataset=dataset)
    # tp.constant.FrameAction(3)


def tec2plt(Folder, InFile, OutFile=None):
    dataset = tp.data.load_tecplot(
        Folder + InFile + '.dat', read_data_option=2)
    if OutFile is None:
        filename = InFile
    else:
        filename = OutFile
    tp.data.save_tecplot_plt(Folder + filename + '.plt', dataset=dataset)


def tec2szplt(Folder, InFile, OutFile=None):
    dataset = tp.data.load_tecplot(
        Folder + InFile + '.dat', read_data_option=2)
    tp.data.save_tecplot_szl(Folder + OutFile + '.szplt', dataset=dataset)


def frame2plt(dataframe,
              SaveFolder,
              OutFile,
              time=None,
              z=None,
              zonename=None,
              float_format='%.8f'):
    frame2tec(dataframe, SaveFolder, OutFile, time, z, zonename, float_format)
    tec2plt(SaveFolder, OutFile, OutFile)


def frame2szplt(dataframe,
                SaveFolder,
                OutFile,
                time=None,
                z=None,
                zonename=None,
                float_format='%.8f'):
    frame2tec(dataframe, SaveFolder, OutFile, time, z, zonename, float_format)
    tec2szplt(SaveFolder, OutFile, OutFile)


# Obtain Spanwise Average Value of Data
def SpanAve(DataFrame, OutputFile = None):
    grouped = DataFrame.groupby(['x', 'y'])
    DataFrame = grouped.mean().reset_index()
    if OutputFile is not None:
        outfile  = open(OutputFile, 'x')
        DataFrame.to_csv(outfile, index=False, sep = '\t')
        outfile.close()


def TimeAve(DataFrame):
    grouped = DataFrame.groupby(['x', 'y', 'z'])
    DataFrame = grouped.mean().reset_index()
    return (DataFrame)


def create_folder(path):
    if not os.path.exists(path + 'Figures'):
        os.mkdir(path + 'Figures')
    exists = os.path.exists(path + 'Instant')
    if not exists:
        os.mkdir(path + 'Instant')
    exists = os.path.exists(path + 'MeanFlow')
    if not exists:
        os.mkdir(path + 'MeanFlow')
    exists = os.path.exists(path + 'probes')
    if not exists:
        os.mkdir(path + 'probes')
    exists = os.path.exists(path + 'Slice')
    if not exists:
        os.mkdir(path + 'Slice')
    exists = os.path.exists(path + 'TimeAve')
    if not exists:
        os.mkdir(path + 'TimeAve')
    exists = os.path.exists(path + 'Vortex')
    if not exists:
        os.mkdir(path + 'Vortex')
    exists = os.path.exists(path + 'snapshots')
    if not exists:
        os.mkdir(path + 'snapshots')
    pathF = path + 'Figures/'
    pathP = path + "probes/"
    pathM = path + "MeanFlow/"
    pathS = path + "Slice/"
    pathT = path + "TimeAve/"
    pathI = path + "Instant/"
    return(pathF, pathP, pathM, pathS, pathT, pathI)



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

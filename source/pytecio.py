# -*- coding: utf-8 -*-
"""
Created on Mon Aug 1 10:24:50 2022
    This code for reading binary data from tecplot (.plt) and
    Convert data to pandas dataframe using tecio
@author: Weibo Hu
"""
# %% load module
import ctypes
import numpy as np
import sys
from pathlib import Path
import os
import pandas as pd
# %% load dll or so library
FieldDataType_Double = 2
FieldDataType_Float  = 1
FieldDataType_Int32 = 3 # -100:not defined
FieldDataType_Int16 = -100 
FieldDataType_Byte = -100
Structed_Grid = 0

def get_dll():
    if sys.platform.startswith('win'):
        dll_path = '../lib/tecio.dll'

    elif sys.platform.startswith('linux'):
        dll_path = '../lib/tecio.so'

    if not os.path.exists(dll_path):
        sys.exit('cannot find tecio library!')

    return ctypes.cdll.LoadLibrary(dll_path)
GLOBAL_DLL = get_dll()
# %%     
class zone_data(dict):
    def __init__(self,parent,zone_n):
        super().__init__()
        self.parent = parent
        self.zone_n = zone_n 
        self.update({i:None for i in parent.nameVars})
    def __getitem__(self,key):
        if isinstance(key, int):
            key = self.parent.nameVars[key]
        t = super().__getitem__(key)
        if t is None:
            var_n = self.parent.nameVars_dict[key] + 1
            t = self.parent._read_zone_var(self.zone_n, var_n)
            self[key] = t
            return t
        else:
            return t
    def __setitem__(self,key,value):
        if isinstance(key, int):
            key = self.parent.nameVars[key]
        if key not in self.parent.nameVars:
            self.parent._add_variable(self.zone_n,key,value)
        super().__setitem__(key,value)
    def __getattr__(self,attr):
        if attr == 'Elements':
            self.Elements = self.parent._retrieve_zone_node_map(self.zone_n)
            return self.Elements
        else:
            raise Exception('no attribute {}'.format(attr))
#zone_n:the number of zones, start from 1 to end, var_n is the same


class SzpltData(dict):  
    def __init__(self, filename, isread=False):
        super().__init__()
        if not isinstance(filename,str):
            self.GenerateDataFromOtherFormat(filename)
            return
        self.dll = GLOBAL_DLL
        
        self.filename = filename
        self.added_new_zone = False
        self.filehandle = self._get_filehandle()
        self.title = self._tecDataSetGetTitle()
        self.numVars = self._tecDataSetGetNumVars()
        self.nameVars = self._tecVarGetName()

        self.fileType = self._tecFileGetType()
        self.numZones = self._tecDataSetGetNumZones()
        self.nameZones = self._tecZoneGetTitle()

        self.nameZones_dict = {k:i for i,k in enumerate(self.nameZones)}
        self.nameVars_dict = {k:i for i,k in enumerate(self.nameVars)}
        
        def cal_zone(i, zone_name):
            d = dict()
            d['varTypes'] = [self._tecZoneVarGetType(i+1,j+1) for j in range(self.numVars)]
            d['passiveVarList'] = [self._tecZoneVarIsPassive(i+1,j+1) for j in range(self.numVars)]
            d['shareVarFromZone'] = [self._tecZoneVarGetSharedZone(i+1,j+1) for j in range(self.numVars)]
            # valueLocation: value 1 represent the data is saved on nodes, value 0 means on elements center
            d['valueLocation'] = [self._tecZoneVarGetValueLocation(i+1,j+1) for j in range(self.numVars)]
            d['IJK'] = self._tecZoneGetIJK(i+1)
            d['zoneType'] = self._tecZoneGetType(i+1)
            d['solutionTime'] = self._tecZoneGetSolutionTime(i+1)
            d['strandID'] = self._tecZoneGetStrandID(i+1)
            d['shareConnectivityFromZone'] = self._tecZoneConnectivityGetSharedZone(i+1)
            d['faceNeighborMode'] = self._tecZoneFaceNbrGetMode(i+1)
            d['numFaceConnections'] = self._tecZoneFaceNbrGetNumConnections(i+1)
            if d['numFaceConnections'] > 0:
                d['faceConnections'] = self._tecZoneFaceNbrGetConnections(i+1)
            d['parentZone'] = self._tecZoneGetParentZone(i+1)
            d['name'] = zone_name
            d['aux'] = self._retrieve_aux_data(i+1)
            return d
        
        self.zone_info = [cal_zone(i,zone_name) for i,zone_name in enumerate(self.nameZones)]
        self.update({name:zone_data(self,i+1) for i,name in enumerate(self.nameZones)})
        # self._retrieve_zone_node_map(1)
        # self._retrieve_aux_data(1)
        if isread:
            [zone[var_name] for zone in self.values() for var_name in self.nameVars]

    def __getitem__(self,key):
        if isinstance(key, int):
            key = self.nameZones[key]
        return super().__getitem__(key)
    def __setitem__(self,key,value):
        self.added_new_zone = True 
        return super().__setitem__(key,value)
    def _read_zone_var(self,zone_n,var_n):
        
        info = self.zone_info[zone_n - 1]
        numValues = self._tecZoneVarGetNumValues(zone_n, var_n)

        if info['passiveVarList'][var_n - 1] == 0:
            fieldDataType = info['varTypes'][var_n-1]
            if fieldDataType == FieldDataType_Float:
                d = self._get_data_all_type(zone_n, var_n, numValues, ctypes.c_float, self.dll.tecZoneVarGetFloatValues)
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Double:
                d = self._get_data_all_type(zone_n, var_n, numValues, ctypes.c_double, self.dll.tecZoneVarGetDoubleValues)
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Int32:
                d = self._get_data_all_type(zone_n, var_n, numValues, ctypes.c_int, self.dll.tecZoneVarGetInt32Values)
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Int16:
                d = self._get_data_all_type(zone_n, var_n, numValues, ctypes.c_int, self.dll.tecZoneVarGetInt16Values)
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Byte:
                d = self._get_data_all_type(zone_n, var_n, numValues, ctypes.c_int, self.dll.tecZoneVarGetByteValues)
                # np_array = np.array(d)
            else:
                raise Exception('FieldDataType Error:not defined data type')
            d = np.array(d)
            # if info['zoneType'] == Structed_Grid: # structed grid
                # Imax,Jmax,Kmax = info['IJK'] 
                # if d.size != Imax*Jmax*Kmax:
                #     Imax =max(Imax - 1,1)
                #     Jmax =max(Jmax - 1,1)
                #     Kmax =max(Kmax - 1,1)
                # d = d.reshape((Kmax,Jmax,Imax)).transpose((2,1,0))
            return d
        else:
            return np.array([])
    def _get_data_all_type(self, zone_n, var_n, numValues, c_type, fun):
        t = (c_type*numValues)()
        fun(self.filehandle, zone_n, var_n, 1, numValues, t)
        return t
    def _get_filehandle(self):
        '''get the filehandle'''
        p = ctypes.c_int(13)
        p1 = ctypes.pointer(p)
        filehandle = ctypes.pointer(p1)
        name = ctypes.c_char_p(self.filename.encode())
        self.dll.tecFileReaderOpen(name,filehandle)
        return filehandle[0]
    def _tecDataSetGetTitle(self):
        '''get the title of data set'''
        s = ctypes.c_char_p()
        ll = ctypes.pointer(s)
        self.dll.tecDataSetGetTitle(self.filehandle,ll)
        t = ll[0].decode()

        return t
    def _tecDataSetGetNumVars(self):
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        self.dll.tecDataSetGetNumVars(self.filehandle,p)
        return p[0]
    def _tecVarGetName(self):
        def get_name(i):
            s = ctypes.c_char_p()
            ll = ctypes.pointer(s)
            self.dll.tecVarGetName(self.filehandle,i,ll)
            return ll[0].decode()
        name_list = [get_name(i) for i in range(1,self.numVars+1)]

        return name_list
    def _tecFileGetType(self):
        '''获取文件类型，即数据存储的格式在写文件的时候可以用到'''
        s = ctypes.c_int(-100)
        ll = ctypes.pointer(s)
        self.dll.tecFileGetType(self.filehandle,ll)
        t = ll[0]

        return t
    def _tecDataSetGetNumZones(self):
        '''获取数据总共包含的zone的个数'''
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        self.dll.tecDataSetGetNumZones(self.filehandle,p)

        return p[0]
    def _tecZoneGetTitle(self):
        '''获取每个zone的名字'''
        def get_name(i):
            s = ctypes.c_char_p()
            ll = ctypes.pointer(s)
            self.dll.tecZoneGetTitle(self.filehandle,i,ll)
            return ll[0].decode()
        name_list = [get_name(i) for i in range(1,self.numZones+1)]

        return name_list
    def _tecZoneVarGetType(self,zone_n,var_n):
        '''获取数据存储的类型 是double（64） 还是single（32）double型返回True'''
        p = self._return_2_int(zone_n,var_n,self.dll.tecZoneVarGetType)
        #if p is FieldDataType_Double, it is double format
        return p
        
    def _tecZoneVarGetSharedZone(self,zone_n,var_n):
        '''    '''
        return self._return_2_int(zone_n,var_n,self.dll.tecZoneVarGetSharedZone)
    
    def _tecZoneVarGetValueLocation(self,zone_n,var_n):
        '''    '''
        return self._return_2_int(zone_n,var_n,self.dll.tecZoneVarGetValueLocation)
    
    def _tecZoneVarIsPassive(self,zone_n,var_n):
        '''    '''
        return self._return_2_int(zone_n, var_n, self.dll.tecZoneVarIsPassive)
    
    def _return_1_int(self,n,fun):
        '''执行fun(filehandle,int,&int)函数并返回结果'''
        p = ctypes.pointer(ctypes.c_int(0))
        fun(self.filehandle,n,p)
        return p[0]

    def _add_variable(self,zone_n,var_name,value):
        ''' add a new variable to all zones'''
        info = self.zone_info[zone_n -1]
        self.nameVars.append(var_name)
        self.nameVars_dict[var_name] = len(self.nameVars) - 1
        info['varTypes'].append(info['varTypes'][-1])
        info['shareVarFromZone'].append(0)
        I,J,K = info['IJK']
        if info['zoneType'] == Structed_Grid:#structed IJK type
            if value.size == I*J*K:
                valueLocation = 1
            else:
                valueLocation = 0
        else:
            if value.size == I:
                valueLocation = 1
            else:
                valueLocation = 0
        info['valueLocation'].append(valueLocation)
        info['passiveVarList'].append(0)
        for zone_p, item in enumerate(self.zone_info):

            if zone_n == zone_p+1:
                continue
            else:
                item['varTypes'].append(item['varTypes'][-1])
                item['shareVarFromZone'].append(0)
                item['valueLocation'].append(valueLocation)
                item['passiveVarList'].append(1)
        for zone_data_ in self.values():
            zone_data_[var_name] = None
    def _return_2_int(self,zone_n,var_n,fun):
        '''执行fun(filehandle,int,int,&int)函数并返回结果'''
        p = ctypes.pointer(ctypes.c_int(0))
        fun(self.filehandle,zone_n,var_n,p)
        return p[0]
    def _return_n_array(self,fun,c_type, numValues,*d):
        '''输入参数是n个整数，返回长为numValues的c_type类型的一个数组并转化为ndarry'''
        t = (c_type*numValues)()
        fun(self.filehandle, *d, t)
        return np.array(t)
    def _tecZoneGetType(self,zone_n):
        '''获取zone的类型'''
        t = self._return_1_int(zone_n,self.dll.tecZoneGetType)
        if t == 6 or t == 7:
            raise Exception('Unsupported zone type')

        return t
    
    def _tecZoneGetIJK(self,zone_n):
        '''获取该zone 的ijk的值'''
        iMax = ctypes.pointer(ctypes.c_int(0))
        jMax = ctypes.pointer(ctypes.c_int(0))
        kMax = ctypes.pointer(ctypes.c_int(0))
        self.dll.tecZoneGetIJK(self.filehandle,zone_n,iMax,jMax,kMax)
        t = iMax[0], jMax[0], kMax[0]
 
        return t
    def _tecZoneConnectivityGetSharedZone(self,zone_n):
        shareConnectivityFromZone = self._return_1_int(zone_n,self.dll.tecZoneConnectivityGetSharedZone)
        return shareConnectivityFromZone

    def _tecZoneFaceNbrGetMode(self,zone_n):
        faceNeighborMode = self._return_1_int(zone_n,self.dll.tecZoneFaceNbrGetMode)
        return faceNeighborMode

    def _tecZoneFaceNbrGetNumConnections(self,zone_n):
        numFaceConnections = self._return_1_int(zone_n,self.dll.tecZoneFaceNbrGetNumConnections)
        return numFaceConnections
    def _tecZoneFaceNbrGetConnections(self,zone_n):
        numFaceValues = self._return_1_int(zone_n,self.dll.tecZoneFaceNbrGetNumValues)
        are64Bit = self._return_1_int(zone_n,self.dll.tecZoneFaceNbrsAre64Bit)
        if are64Bit:
            faceConnections = self._return_n_array(self.dll.tecZoneFaceNbrGetConnections64,
                                                    ctypes.c_long,numFaceValues,zone_n)
        else:
            faceConnections = self._return_n_array(self.dll.tecZoneFaceNbrGetConnections,
                                                       ctypes.c_int,numFaceValues,zone_n)
        return faceConnections
    def _tecZoneGetSolutionTime(self,zone_n):
        d = ctypes.c_double(0.0)
        p = ctypes.pointer(d)
        self.dll.tecZoneGetSolutionTime(self.filehandle,zone_n,p)
        solutionTime = p[0]

        return solutionTime

    def _tecZoneGetStrandID(self,zone_n):
        StrandID = self._return_1_int(zone_n,self.dll.tecZoneGetStrandID)

        return StrandID

    def _tecZoneGetParentZone(self,zone_n):
        parentZone = self._return_1_int(zone_n,self.dll.tecZoneGetParentZone)

        return parentZone

    def _tecZoneVarGetNumValues(self,zone_n,var_n):
        numValues = self._return_2_int(zone_n,var_n,self.dll.tecZoneVarGetNumValues)

        return numValues

    def _tecZoneFaceNbrGetNumValues(self,zone_n):
        k = self._return_1_int(zone_n,self.dll.tecZoneFaceNbrGetNumValues)

        return k
    def _retrieve_zone_node_map(self,zone_n):
        info = self.zone_info[zone_n-1]
        if info['zoneType'] != Structed_Grid and info['shareConnectivityFromZone'] == 0:
            jMax = info['IJK'][1]
            numValues = self._tecZoneNodeMapGetNumValues(zone_n,jMax)

            is64Bit = self._tecZoneNodeMapIs64Bit(zone_n)
            if is64Bit != 0:
                #is64bit True
                nodeMap = self._return_n_array(self.dll.tecZoneNodeMapGet64, ctypes.c_long, numValues, zone_n,1,jMax)
            else:
                nodeMap = self._return_n_array(self.dll.tecZoneNodeMapGet, ctypes.c_int, numValues, zone_n,1,jMax)

        return nodeMap.reshape((jMax,-1))
    def _retrieve_aux_data(self,zone_n):
        numItems = self._tecZoneAuxDataGetNumItems(zone_n)

        if numItems!=0:
            aux_data = dict()
            for whichItem in range(1,numItems+1):
                name = ctypes.c_char_p()
                value = ctypes.c_char_p()
                name_p = ctypes.pointer(name)
                value_p = ctypes.pointer(value)
                self.dll.tecZoneAuxDataGetItem(self.filehandle,zone_n,whichItem,name_p,value_p)
                name = name_p[0].decode()
                value = value_p[0].decode()
                aux_data[name]=value
            return aux_data
        else:
            return None
    def _tecZoneAuxDataGetNumItems(self,zone_n):
        return self._return_1_int(zone_n,self.dll.tecZoneAuxDataGetNumItems)

    def _retrieve_custom_label_sets(self,zone_n):
        pass

    def _tecCustomLabelsGetNumSets(self,zone_n):
        return self._return_1_int(zone_n,self.dll.tecCustomLabelsGetNumSets)

    def _tecZoneNodeMapGetNumValues(self,zone_n,jmax):
        return self._return_2_int(zone_n,jmax,self.dll.tecZoneNodeMapGetNumValues)
    
    def _tecZoneNodeMapIs64Bit(self, zone_n):
        return self._return_1_int(zone_n,self.dll.tecZoneNodeMapIs64Bit)

    def close(self):
        self.dll.tecFileReaderClose(ctypes.pointer(self.filehandle))
    
    def write(self,filename,verbose = True):
        k = write_tecio(filename,self,verbose=verbose)
        k.close()
    
    def judge_valuelocation_passive(self,zone_name,var_name,name0):
        I,J,K = self[zone_name][name0].shape 
        value = self[zone_name][var_name]
        # print(zone_name,var_name,value is None)
        if value is None:
            return var_name, 1, 1, 'float32'
        if self.Unstructed:
            if value.size == I:
                valueLocation = 1
            else:
                valueLocation = 0
        else:
            #Structed_grid
            if value.size == I*J*K:
                valueLocation = 1
            else:
                valueLocation = 0
        return var_name, valueLocation, 0, str(value.dtype)
    def sort_nameVars(self):
        def fun_key(name):
            if name.find('Coordinate') != -1:
                return ord(name[-1])
            if name.lower() in 'xyz':
                return 256 + ord(name)
            return sum([ord(i) for i in name]) + 500
        self.nameVars.sort(key = fun_key)
    def judge_unstructed(self,dataset):
        self.Unstructed = False
        for i in dataset.values():
            for j in i.values():
                shape = j.shape
                if j.ndim>1:
                    if shape[1]*shape[2] > 1:
                        self.Unstructed = False 
                        return
                else: 
                    self.Unstructed = True 
                    return
    def GenerateDataFromOtherFormat(self,dataset):
        #将其他类型的数据转化为SzpltData类型
        if isinstance(dataset,SzpltData):
            self = SzpltData
            return
        elif isinstance(dataset,list) or isinstance(dataset,tuple):
            dataset = {str(i+1):v for i,v in enumerate(dataset)}

        aux_data = []
        for v in dataset.values():
            for j in v.keys():
                if not isinstance(v[j],np.ndarray):
                    aux_data.append(j)
            break
        dataset = {i:{j:vd for j,vd in v.items() if j not in aux_data} for i,v in dataset.items()}
        self.judge_unstructed(dataset)
        self.update(dataset)

        self.nameZones = list(self.keys())
        name0 = list(self[self.nameZones[0]].keys())[0]
        loc_pass = [self.judge_valuelocation_passive(zone,vname,name0) for zone in self.keys() for vname in self[zone].keys()]
        loc_pass = set(loc_pass)
        loc_pass_name = set([i[:3] for i in loc_pass])
        self.nameVars = [i[0] for i in loc_pass_name]
        assert len(set(self.nameVars)) == len(loc_pass_name)
        nameVars_ = list(self[self.nameZones[0]].keys())
        for i in self.nameVars:
            if i not in nameVars_:
                nameVars_.append(i)
        self.nameVars = nameVars_
        self.sort_nameVars()
        empty = np.array([])
        for zone_name_,zone in self.items():
            I,J,K = zone[name0].shape
            for var_name,location,passive,dtype in loc_pass:
                if var_name not in zone:
                    if passive == 0:
                        if not self.Unstructed:
                            if location == 1:
                                t = np.zeros((I,J,K),dtype=dtype)
                            else:
                                t = np.zeros((I-1,J-1,K-1),dtype=dtype)
                        else:
                            if location == 1:
                                t = np.zeros((I,J,K),dtype=dtype)
                            else:
                                print(zone_name_,var_name)
                                raise Exception("Unstructed grid center value")
                    else:
                        t = empty
                    zone[var_name] = t
        self.title = 'Pytecio data'
        def cal_zone_info(name_zone,value_location):
            d = dict()
            zone_value = self[name_zone]
            empty = np.array([])
            shape = self[name_zone][self.nameVars[0]].shape
            zoneType = Structed_Grid
            if len(shape) == 1:
                shape = shape[0],1,1
                zoneType = 1
            elif len(shape)==2:
                shape = 1,shape[0],shape[1]
            d['varTypes'] = [self.get_varTypes(name_zone,j) for j in self.nameVars]
            d['passiveVarList'] = [0 if zone_value.get(i,empty).size>0 else 1 for i in self.nameVars]
            d['shareVarFromZone'] = [0] * len(self.nameVars)
            # valueLocation: value 1 represent the data is saved on nodes, value 0 means on elements center
            d['valueLocation'] = value_location
            d['IJK'] = shape
            d['zoneType'] = zoneType
            d['solutionTime'] = .0
            d['strandID'] = 0
            d['shareConnectivityFromZone'] = 0
            d['faceNeighborMode'] = 0
            d['numFaceConnections'] = 0
            d['parentZone'] = 0
            d['name'] = name_zone
            return d
        temp_zone = self[self.nameZones[0]]
        value_location = [sum(temp_zone[key].shape) for key in self.nameVars]
        max_location = max(value_location)
        value_location = [0 if i<max_location else 1 for i in value_location]
        self.zone_info = [cal_zone_info(i,value_location) for i in self.nameZones]
        self.fileType = 0
        self.added_new_zone = False
    def get_varTypes(self,name_zone,name_var):
        varTypes={'int32':3,'float64':2,'float32':1}
        d = self[name_zone][name_var]
        dtype = str(d.dtype)
        if dtype == 'int64':
            d = d.astype('int32')
            self[name_zone][name_var] = d
            dtype = 'int32'
        return varTypes[dtype]

class write_tecio:
    fileFormat = 0 #.szplt

    def __init__(self,filename,dataset=None ,verbose = True):
        '''
        dataset 只要是包含两层字典的数据都可以 like d[key_zone][key_var],如果是非SzpltData类型的数据，目前只支持结构化的数据
        '''
        self.filename = filename
        self.verbose = verbose
        if hasattr(dataset,'added_new_zone') and dataset.added_new_zone:
            dataset = {k:{k2:dataset[k][k2] for k2 in dataset[k].keys()} for k in dataset.keys()}
        if not isinstance(dataset,SzpltData):
            dataset = SzpltData(dataset)
        self.dataset = dataset
        self.dll = GLOBAL_DLL
        self.filehandle = self._get_filehandle()
        empty = np.array([])
        for i,zone_name in enumerate(dataset.nameZones):
            info = dataset.zone_info[i]
            I,J,K = info['IJK']
            zone_set = dataset[zone_name]
            varTypes = self._list_to_int_array(info['varTypes'])

            #因为这里有个bug所以我加了这样一句转化,原因是第一个zone共享了第一个zone 在创建的时候会导致失败，所以在写文件时强制取消shared
            shareVarFromZone = self._list_to_int_array(info['shareVarFromZone'])
            valueLocation = self._list_to_int_array(info['valueLocation'])
            info['passiveVarList'] = [0 if zone_set.get(i,empty).size>0 else 1 for i in dataset.nameVars]
            passiveVarList = self._list_to_int_array(info['passiveVarList'])
            
            if info['zoneType'] == Structed_Grid:
                outputZone = self._tecZoneCreateIJK(zone_name,I,J,K,varTypes, shareVarFromZone,
                valueLocation, passiveVarList, info['shareConnectivityFromZone'], info['numFaceConnections'], info['faceNeighborMode'])
            else:
                outputZone = self._tecZoneCreateFE(zone_name, info['zoneType'], I, J, varTypes, shareVarFromZone,
                valueLocation, passiveVarList, info['shareConnectivityFromZone'], info['numFaceConnections'], info['faceNeighborMode'])
            
            self._tecZoneSetUnsteadyOptions(outputZone, info['solutionTime'], info['strandID'])
            if info['parentZone'] != 0:
                self._tecZoneSetParentZone(outputZone,info['parentZone'])
            if info['numFaceConnections'] > 0:
                faceConnections = info['faceConnections']
                if isinstance(faceConnections,list) or isinstance(faceConnections,tuple):
                    faceConnections = np.array(faceConnections,dtype='int64')
                    print(faceConnections)
                if faceConnections.itemsize == 8:
                    self._write_data_all_type(self.dll.tecZoneFaceNbrWriteConnections64,
                                              faceConnections.ctypes,outputZone)
                else:
                    self._write_data_all_type(self.dll.tecZoneFaceNbrWriteConnections32,
                                              faceConnections.ctypes,outputZone)
            if info.get('aux') is not None:
                for key,value in info['aux'].items():
                    key_p = ctypes.c_char_p(key.encode())
                    value_p = ctypes.c_char_p(value.encode())
                    self.dll.tecZoneAddAuxData(self.filehandle,outputZone,key_p,value_p)
            
            for j,var_name in enumerate(dataset.nameVars):

                var_n = j+1
                data=zone_set[var_name].copy(order='C')
                if info['zoneType'] is Structed_Grid:
                    if data.ndim == 2:
                        shape = data.shape
                        data.shape = 1,shape[0],shape[1]
                    
                    if data.size > 0:
                        data = data.transpose((2,1,0)).copy()
                ff = [min(i,j) for j in info['shareVarFromZone']]
                if info['passiveVarList'][var_n - 1] is 0 and ff[var_n -1] is 0:
                    
                    fieldDataType = info['varTypes'][var_n-1]
                    if fieldDataType is FieldDataType_Float:
                        self._write_data_all_type(self.dll.tecZoneVarWriteFloatValues, data.ctypes, outputZone, var_n, 0, data.size)
                    elif fieldDataType is FieldDataType_Double:
                        self._write_data_all_type(self.dll.tecZoneVarWriteDoubleValues, data.ctypes, outputZone, var_n, 0, data.size)
                    elif fieldDataType is FieldDataType_Int32:
                        self._write_data_all_type(self.dll.tecZoneVarWriteInt32Values, data.ctypes, outputZone, var_n, 0, data.size)
                    elif fieldDataType is FieldDataType_Int16:
                        self._write_data_all_type(self.dll.tecZoneVarWriteInt16Values, data.ctypes, outputZone, var_n, 0, data.size)
                    elif fieldDataType is FieldDataType_Byte:
                        self._write_data_all_type(self.dll.tecZoneVarWriteByteValues, data.ctypes, outputZone, var_n, 0, data.size)
                    else:
                        print(fieldDataType,'iiiiiiiiiiiii')
                        raise Exception('FieldDataType Error:not defined data type')
                
            self._write_zone_node_map(outputZone, info, zone_set)
    def _write_zone_node_map(self,zone_n,info, zone_set):
        # info = self.dataset.zone_info[self.dataset.nameZones[zone_n-1]]
        if info['zoneType'] is not Structed_Grid and info['shareConnectivityFromZone'] is 0:
            Elements = zone_set.Elements
            numValues = Elements.size
            if Elements.itemsize is 8:
                #is64bit True
                self._write_data_all_type(self.dll.tecZoneNodeMapWrite64, Elements.ctypes, zone_n,0,1,numValues)
            else:
                self._write_data_all_type(self.dll.tecZoneNodeMapWrite32, Elements.ctypes, zone_n,0,1,numValues)

    def _list_to_int_array(self,l):
        t = (ctypes.c_int*len(l))()

        for i,j in enumerate( l):
            t[i] = j
        return t
    def _get_filehandle(self):
        p = ctypes.c_int(13)
        p1 = ctypes.pointer(p)
        filehandle = ctypes.pointer(p1)
        name = ctypes.c_char_p(self.filename.encode())
        fileType = self.dataset.fileType
        name_str = ','.join([str(i) for i in self.dataset.nameVars])
        # name_str
        var_list_str = ctypes.c_char_p(name_str.encode())
        title_str = ctypes.c_char_p(self.dataset.title.encode())
        if self.filename.endswith('.szplt'):
            fileFormat = 1
        else:
            raise Exception('file format error')
        self.dll.tecFileWriterOpen(name,title_str,var_list_str,fileFormat,fileType,2,None,filehandle)
        
        #官方例子中有这么一个东西，看名字叫debug 感觉不用也可以,就是在输出szplt文件时输出一些信息
        if self.verbose:
            outputDebugInfo = 1
            self.dll.tecFileSetDiagnosticsLevel(filehandle[0],outputDebugInfo)

        return filehandle[0]

    def _tecZoneCreateIJK(self,zoneTitle, iMax, jMax, kMax, varTypes, shareVarFromZone,
        valueLocation, passiveVarList, shareConnectivityFromZone, numFaceConnections, faceNeighborMode):
        p = ctypes.pointer(ctypes.c_int(0))
        zone_title = ctypes.c_char_p(zoneTitle.encode())
        self.dll.tecZoneCreateIJK(self.filehandle, zone_title, iMax, jMax, kMax, varTypes,shareVarFromZone,
        valueLocation, passiveVarList, shareConnectivityFromZone, numFaceConnections, faceNeighborMode,p)
        return p[0]
    
    def _tecZoneCreateFE(self,zoneTitle, zoneType, iMax, jMax, varTypes,shareVarFromZone,
        valueLocation, passiveVarList, shareConnectivityFromZone, numFaceConnections, faceNeighborMode):
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        zone_title = ctypes.c_char_p(zoneTitle.encode())

        self.dll.tecZoneCreateFE(self.filehandle, zone_title, zoneType, iMax, jMax, varTypes,shareVarFromZone,
        valueLocation, passiveVarList, shareConnectivityFromZone, numFaceConnections, faceNeighborMode,p)
        return p[0]

    def _tecZoneSetUnsteadyOptions(self,zone_n, solutionTime=0, StrandID=0):
        if solutionTime !=0 or StrandID != 0:
            solutionTime = ctypes.c_double(solutionTime)
            self.dll.tecZoneSetUnsteadyOptions(self.filehandle,zone_n, solutionTime, StrandID)
    
    def _tecZoneSetParentZone(self,zone_n,zone_parent):
        self.dll.tecZoneSetParentZone(self.filehandle,zone_n,zone_parent)

    def _write_data_all_type(self,fun,data, *d):
        fun(self.filehandle, *d, data)
    def close(self):
        self.dll.tecFileWriterClose(ctypes.pointer(self.filehandle))


def read(filename,isread=False):
    return SzpltData(filename,isread)


def write(filename,dataset,verbose = True):
    t = write_tecio(filename,dataset, verbose=verbose)
    t.close()


def cal_zone(number,g,q):
    g = g[number]
    q = q[number]
    k = {i:g[i] for i in 'XYZ'}
    y = {'VAR{}'.format(key):val for key,val in q.items() if isinstance(key,int)}
    k.update(y)
    return k   

# %%
def ReadSinglePlt(file_nm, var_list=None):
    file = SzpltData(filename=file_nm)
    zone_num = file.numZones
    SolTime = file.zone_info[0]['solutionTime']
    df = pd.DataFrame()
    for i in range(np.size(zone_num)):
        zonename = file.nameZones[i]
        ZoneDict = file[zonename]
        temp = pd.DataFrame(ZoneDict)
        df = pd.concat([df, temp])
    if var_list is None:
        df_select = df
    else:
        df_select = df[var_list]
    return(df_select, SolTime)

def ReadMultiPlt(folder, var_list=None):
    FileList = os.listdir(folder)
    df, SolTime = ReadSinglePlt(folder+FileList[0], var_list)
    for i in range(np.size(FileList)-1):
        df1, _ = ReadSinglePlt(folder+FileList[i+1], var_list)
        df = pd.concat([df, df1])
    return(df, SolTime)

if __name__ == '__main__':
    path = 'e:/cases/large_oblique/TP_data_00135989/'
    finm = path + 'TP_dat_000001.szplt'
    var_list = ['x', 'y', 'z', 'u']
    df, s_time = ReadSinglePlt(finm, var_list)
    df, s_time = ReadMultiPlt(path, var_list)

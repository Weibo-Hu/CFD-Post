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
import os
import pandas as pd
from glob import glob
from timer import timer

# %% load dll or so library
FieldDataType_Double = 2
FieldDataType_Float = 1
FieldDataType_Int32 = 3  # -100:not defined
FieldDataType_Int16 = -100
FieldDataType_Byte = -100
Structed_Grid = 0
# dll_file = "/mnt/data/BaiduNetdiskWorkspace/Program/tecio/teciosrc/0/libtecio.so"


def get_dll():
    cwd = os.getcwd()
    dll_path = os.path.dirname(cwd) + os.path.sep + "lib" + os.path.sep
    if sys.platform.startswith("win"):
        dll_file = dll_path + "libtecio.dll"
    elif sys.platform.startswith("linux"):
        dll_file = dll_path + "2017r2_tecio.so"
    if not os.path.exists(dll_path):
        sys.exit("cannot find tecio library!")
    return ctypes.cdll.LoadLibrary(dll_file)


GLOBAL_DLL = get_dll()

class zone_data(dict):
    def __init__(self, parent, zone_n):
        super().__init__()
        self.parent = parent
        self.zone_n = zone_n
        self.update({i: None for i in parent.nameVars})

    def __getitem__(self, key):
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

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = self.parent.nameVars[key]
        if key not in self.parent.nameVars:
            self.parent._add_variable(self.zone_n, key, value)
        super().__setitem__(key, value)

    def __getattr__(self, attr):
        if attr == "Elements":
            self.Elements = self.parent._retrieve_zone_node_map(self.zone_n)
            return self.Elements
        else:
            raise Exception("no attribute {}".format(attr))


# zone_n:the number of zones, start from 1 to end, var_n is the same


class SzpltData(dict):
    def __init__(self, filename, isread=False):
        super().__init__()
        if not isinstance(filename, str):
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

        self.nameZones_dict = {k: i for i, k in enumerate(self.nameZones)}
        self.nameVars_dict = {k: i for i, k in enumerate(self.nameVars)}

        def cal_zone(i, zone_name):
            d = dict()
            d["varTypes"] = [
                self._tecZoneVarGetType(i + 1, j + 1) for j in range(self.numVars)
            ]
            d["passiveVarList"] = [
                self._tecZoneVarIsPassive(i + 1, j + 1) for j in range(self.numVars)
            ]
            d["shareVarFromZone"] = [
                self._tecZoneVarGetSharedZone(i + 1, j + 1) for j in range(self.numVars)
            ]
            # valueLocation: value 1 represent the data is saved on nodes, value 0 means on elements center
            d["valueLocation"] = [
                self._tecZoneVarGetValueLocation(i + 1, j + 1)
                for j in range(self.numVars)
            ]
            d["IJK"] = self._tecZoneGetIJK(i + 1)
            d["zoneType"] = self._tecZoneGetType(i + 1)
            d["solutionTime"] = self._tecZoneGetSolutionTime(i + 1)
            d["strandID"] = self._tecZoneGetStrandID(i + 1)
            d["shareConnectivityFromZone"] = self._tecZoneConnectivityGetSharedZone(
                i + 1
            )
            d["faceNeighborMode"] = self._tecZoneFaceNbrGetMode(i + 1)
            d["numFaceConnections"] = self._tecZoneFaceNbrGetNumConnections(
                i + 1)
            if d["numFaceConnections"] > 0:
                d["faceConnections"] = self._tecZoneFaceNbrGetConnections(
                    i + 1)
            d["parentZone"] = self._tecZoneGetParentZone(i + 1)
            d["name"] = zone_name
            d["aux"] = self._retrieve_aux_data(i + 1)
            return d

        self.zone_info = [
            cal_zone(i, zone_name) for i, zone_name in enumerate(self.nameZones)
        ]
        self.update(
            {name: zone_data(self, i + 1)
             for i, name in enumerate(self.nameZones)}
        )
        # self._retrieve_zone_node_map(1)
        # self._retrieve_aux_data(1)
        if isread:
            [zone[var_name] for zone in self.values()
             for var_name in self.nameVars]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.nameZones[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.added_new_zone = True
        return super().__setitem__(key, value)

    def _read_zone_var(self, zone_n, var_n):

        info = self.zone_info[zone_n - 1]
        numValues = self._tecZoneVarGetNumValues(zone_n, var_n)

        if info["passiveVarList"][var_n - 1] == 0:
            fieldDataType = info["varTypes"][var_n - 1]
            if fieldDataType == FieldDataType_Float:
                d = self._get_data_all_type(
                    zone_n,
                    var_n,
                    numValues,
                    ctypes.c_float,
                    self.dll.tecZoneVarGetFloatValues,
                )
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Double:
                d = self._get_data_all_type(
                    zone_n,
                    var_n,
                    numValues,
                    ctypes.c_double,
                    self.dll.tecZoneVarGetDoubleValues,
                )
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Int32:
                d = self._get_data_all_type(
                    zone_n,
                    var_n,
                    numValues,
                    ctypes.c_int,
                    self.dll.tecZoneVarGetInt32Values,
                )
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Int16:
                d = self._get_data_all_type(
                    zone_n,
                    var_n,
                    numValues,
                    ctypes.c_int,
                    self.dll.tecZoneVarGetInt16Values,
                )
                # np_array = np.array(d)
            elif fieldDataType == FieldDataType_Byte:
                d = self._get_data_all_type(
                    zone_n,
                    var_n,
                    numValues,
                    ctypes.c_int,
                    self.dll.tecZoneVarGetByteValues,
                )
                # np_array = np.array(d)
            else:
                raise Exception("FieldDataType Error:not defined data type")
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
        t = (c_type * numValues)()
        fun(self.filehandle, zone_n, var_n, 1, numValues, t)
        return t

    def _get_filehandle(self):
        """get the filehandle"""
        p = ctypes.c_int(13)
        p1 = ctypes.pointer(p)
        filehandle = ctypes.pointer(p1)
        name = ctypes.c_char_p(self.filename.encode())
        self.dll.tecFileReaderOpen(name, filehandle)
        return filehandle[0]

    def _tecDataSetGetTitle(self):
        """get the title of data set"""
        s = ctypes.c_char_p()
        ll = ctypes.pointer(s)
        self.dll.tecDataSetGetTitle(self.filehandle, ll)
        t = ll[0].decode()

        return t

    def _tecDataSetGetNumVars(self):
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        self.dll.tecDataSetGetNumVars(self.filehandle, p)
        return p[0]

    def _tecVarGetName(self):
        def get_name(i):
            s = ctypes.c_char_p()
            ll = ctypes.pointer(s)
            self.dll.tecVarGetName(self.filehandle, i, ll)
            return ll[0].decode()

        name_list = [get_name(i) for i in range(1, self.numVars + 1)]

        return name_list

    def _tecFileGetType(self):
        """获取文件类型，即数据存储的格式在写文件的时候可以用到"""
        s = ctypes.c_int(-100)
        ll = ctypes.pointer(s)
        self.dll.tecFileGetType(self.filehandle, ll)
        t = ll[0]

        return t

    def _tecDataSetGetNumZones(self):
        """获取数据总共包含的zone的个数"""
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        self.dll.tecDataSetGetNumZones(self.filehandle, p)

        return p[0]

    def _tecZoneGetTitle(self):
        """获取每个zone的名字"""

        def get_name(i):
            s = ctypes.c_char_p()
            ll = ctypes.pointer(s)
            self.dll.tecZoneGetTitle(self.filehandle, i, ll)
            return ll[0].decode()

        name_list = [get_name(i) for i in range(1, self.numZones + 1)]

        return name_list

    def _tecZoneVarGetType(self, zone_n, var_n):
        """获取数据存储的类型 是double（64） 还是single（32）double型返回True"""
        p = self._return_2_int(zone_n, var_n, self.dll.tecZoneVarGetType)
        # if p is FieldDataType_Double, it is double format
        return p

    def _tecZoneVarGetSharedZone(self, zone_n, var_n):
        """    """
        return self._return_2_int(zone_n, var_n, self.dll.tecZoneVarGetSharedZone)

    def _tecZoneVarGetValueLocation(self, zone_n, var_n):
        """    """
        return self._return_2_int(zone_n, var_n, self.dll.tecZoneVarGetValueLocation)

    def _tecZoneVarIsPassive(self, zone_n, var_n):
        """    """
        return self._return_2_int(zone_n, var_n, self.dll.tecZoneVarIsPassive)

    def _return_1_int(self, n, fun):
        """执行fun(filehandle,int,&int)函数并返回结果"""
        p = ctypes.pointer(ctypes.c_int(0))
        fun(self.filehandle, n, p)
        return p[0]

    def _add_variable(self, zone_n, var_name, value):
        """ add a new variable to all zones"""
        info = self.zone_info[zone_n - 1]
        self.nameVars.append(var_name)
        self.nameVars_dict[var_name] = len(self.nameVars) - 1
        info["varTypes"].append(info["varTypes"][-1])
        info["shareVarFromZone"].append(0)
        I, J, K = info["IJK"]
        if info["zoneType"] == Structed_Grid:  # structed IJK type
            if value.size == I * J * K:
                valueLocation = 1
            else:
                valueLocation = 0
        else:
            if value.size == I:
                valueLocation = 1
            else:
                valueLocation = 0
        info["valueLocation"].append(valueLocation)
        info["passiveVarList"].append(0)
        for zone_p, item in enumerate(self.zone_info):

            if zone_n == zone_p + 1:
                continue
            else:
                item["varTypes"].append(item["varTypes"][-1])
                item["shareVarFromZone"].append(0)
                item["valueLocation"].append(valueLocation)
                item["passiveVarList"].append(1)
        for zone_data_ in self.values():
            zone_data_[var_name] = None

    def _return_2_int(self, zone_n, var_n, fun):
        """执行fun(filehandle,int,int,&int)函数并返回结果"""
        p = ctypes.pointer(ctypes.c_int(0))
        fun(self.filehandle, zone_n, var_n, p)
        return p[0]

    def _return_n_array(self, fun, c_type, numValues, *d):
        """输入参数是n个整数，返回长为numValues的c_type类型的一个数组并转化为ndarry"""
        t = (c_type * numValues)()
        fun(self.filehandle, *d, t)
        return np.array(t)

    def _tecZoneGetType(self, zone_n):
        """获取zone的类型"""
        t = self._return_1_int(zone_n, self.dll.tecZoneGetType)
        if t == 6 or t == 7:
            raise Exception("Unsupported zone type")
        return t

    def _tecZoneGetIJK(self, zone_n):
        """获取该zone 的ijk的值"""
        iMax = ctypes.pointer(ctypes.c_int(0))
        jMax = ctypes.pointer(ctypes.c_int(0))
        kMax = ctypes.pointer(ctypes.c_int(0))
        self.dll.tecZoneGetIJK(self.filehandle, zone_n, iMax, jMax, kMax)
        t = iMax[0], jMax[0], kMax[0]

        return t

    def _tecZoneConnectivityGetSharedZone(self, zone_n):
        shareConnectivityFromZone = self._return_1_int(
            zone_n, self.dll.tecZoneConnectivityGetSharedZone
        )
        return shareConnectivityFromZone

    def _tecZoneFaceNbrGetMode(self, zone_n):
        faceNeighborMode = self._return_1_int(
            zone_n, self.dll.tecZoneFaceNbrGetMode)
        return faceNeighborMode

    def _tecZoneFaceNbrGetNumConnections(self, zone_n):
        numFaceConnections = self._return_1_int(
            zone_n, self.dll.tecZoneFaceNbrGetNumConnections
        )
        return numFaceConnections

    def _tecZoneFaceNbrGetConnections(self, zone_n):
        numFaceValues = self._return_1_int(
            zone_n, self.dll.tecZoneFaceNbrGetNumValues)
        are64Bit = self._return_1_int(zone_n, self.dll.tecZoneFaceNbrsAre64Bit)
        if are64Bit:
            faceConnections = self._return_n_array(
                self.dll.tecZoneFaceNbrGetConnections64,
                ctypes.c_long,
                numFaceValues,
                zone_n,
            )
        else:
            faceConnections = self._return_n_array(
                self.dll.tecZoneFaceNbrGetConnections,
                ctypes.c_int,
                numFaceValues,
                zone_n,
            )
        return faceConnections

    def _tecZoneGetSolutionTime(self, zone_n):
        d = ctypes.c_double(0.0)
        p = ctypes.pointer(d)
        self.dll.tecZoneGetSolutionTime(self.filehandle, zone_n, p)
        solutionTime = p[0]

        return solutionTime

    def _tecZoneGetStrandID(self, zone_n):
        StrandID = self._return_1_int(zone_n, self.dll.tecZoneGetStrandID)

        return StrandID

    def _tecZoneGetParentZone(self, zone_n):
        parentZone = self._return_1_int(zone_n, self.dll.tecZoneGetParentZone)

        return parentZone

    def _tecZoneVarGetNumValues(self, zone_n, var_n):
        numValues = self._return_2_int(
            zone_n, var_n, self.dll.tecZoneVarGetNumValues)

        return numValues

    def _tecZoneFaceNbrGetNumValues(self, zone_n):
        k = self._return_1_int(zone_n, self.dll.tecZoneFaceNbrGetNumValues)

        return k

    def _retrieve_zone_node_map(self, zone_n):
        info = self.zone_info[zone_n - 1]
        if info["zoneType"] != Structed_Grid and info["shareConnectivityFromZone"] == 0:
            jMax = info["IJK"][1]
            numValues = self._tecZoneNodeMapGetNumValues(zone_n, jMax)

            is64Bit = self._tecZoneNodeMapIs64Bit(zone_n)
            if is64Bit != 0:
                # is64bit True
                nodeMap = self._return_n_array(
                    self.dll.tecZoneNodeMapGet64,
                    ctypes.c_long,
                    numValues,
                    zone_n,
                    1,
                    jMax,
                )
            else:
                nodeMap = self._return_n_array(
                    self.dll.tecZoneNodeMapGet, ctypes.c_int, numValues, zone_n, 1, jMax
                )
        return nodeMap.reshape((jMax, -1))

    def _retrieve_aux_data(self, zone_n):
        numItems = self._tecZoneAuxDataGetNumItems(zone_n)

        if numItems != 0:
            aux_data = dict()
            for whichItem in range(1, numItems + 1):
                name = ctypes.c_char_p()
                value = ctypes.c_char_p()
                name_p = ctypes.pointer(name)
                value_p = ctypes.pointer(value)
                self.dll.tecZoneAuxDataGetItem(
                    self.filehandle, zone_n, whichItem, name_p, value_p
                )
                name = name_p[0].decode()
                value = value_p[0].decode()
                aux_data[name] = value
            return aux_data
        else:
            return None

    def _tecZoneAuxDataGetNumItems(self, zone_n):
        return self._return_1_int(zone_n, self.dll.tecZoneAuxDataGetNumItems)

    def _retrieve_custom_label_sets(self, zone_n):
        pass

    def _tecCustomLabelsGetNumSets(self, zone_n):
        return self._return_1_int(zone_n, self.dll.tecCustomLabelsGetNumSets)

    def _tecZoneNodeMapGetNumValues(self, zone_n, jmax):
        return self._return_2_int(zone_n, jmax, self.dll.tecZoneNodeMapGetNumValues)

    def _tecZoneNodeMapIs64Bit(self, zone_n):
        return self._return_1_int(zone_n, self.dll.tecZoneNodeMapIs64Bit)

    def close(self):
        self.dll.tecFileReaderClose(ctypes.pointer(self.filehandle))

    def write(self, filename, verbose=True):
        k = write_tecio(filename, self, verbose=verbose)
        k.close()

    def judge_valuelocation_passive(self, zone_name, var_name, name0):
        I, J, K = self[zone_name][name0].shape
        value = self[zone_name][var_name]
        # print(zone_name,var_name,value is None)
        if value is None:
            return var_name, 1, 1, "float32"
        if self.Unstructed:
            if value.size == I:
                valueLocation = 1
            else:
                valueLocation = 0
        else:
            # Structed_grid
            if value.size == I * J * K:
                valueLocation = 1
            else:
                valueLocation = 0
        return var_name, valueLocation, 0, str(value.dtype)

    def sort_nameVars(self):
        def fun_key(name):
            if name.find("Coordinate") != -1:
                return ord(name[-1])
            if name.lower() in "xyz":
                return 256 + ord(name)
            return sum([ord(i) for i in name]) + 500

        self.nameVars.sort(key=fun_key)

    def judge_unstructed(self, dataset):
        self.Unstructed = False
        for i in dataset.values():
            for j in i.values():
                shape = j.shape
                if j.ndim > 1:
                    if shape[1] * shape[2] > 1:
                        self.Unstructed = False
                        return
                else:
                    self.Unstructed = True
                    return

    def GenerateDataFromOtherFormat(self, dataset):
        if isinstance(dataset, SzpltData):
            self = SzpltData
            return
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            dataset = {str(i + 1): v for i, v in enumerate(dataset)}
        aux_data = []
        for v in dataset.values():
            for j in v.keys():
                if not isinstance(v[j], np.ndarray):
                    aux_data.append(j)
            break
        dataset = {
            i: {j: vd for j, vd in v.items() if j not in aux_data}
            for i, v in dataset.items()
        }
        self.judge_unstructed(dataset)
        self.update(dataset)

        self.nameZones = list(self.keys())
        name0 = list(self[self.nameZones[0]].keys())[0]
        loc_pass = [
            self.judge_valuelocation_passive(zone, vname, name0)
            for zone in self.keys()
            for vname in self[zone].keys()
        ]
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
        for zone_name_, zone in self.items():
            I, J, K = zone[name0].shape
            for var_name, location, passive, dtype in loc_pass:
                if var_name not in zone:
                    if passive == 0:
                        if not self.Unstructed:
                            if location == 1:
                                t = np.zeros((I, J, K), dtype=dtype)
                            else:
                                t = np.zeros(
                                    (I - 1, J - 1, K - 1), dtype=dtype)
                        else:
                            if location == 1:
                                t = np.zeros((I, J, K), dtype=dtype)
                            else:
                                print(zone_name_, var_name)
                                raise Exception("Unstructed grid center value")
                    else:
                        t = empty
                    zone[var_name] = t
        self.title = "Pytecio data"

        def cal_zone_info(name_zone, value_location):
            d = dict()
            zone_value = self[name_zone]
            empty = np.array([])
            shape = self[name_zone][self.nameVars[0]].shape
            zoneType = Structed_Grid
            if len(shape) == 1:
                shape = shape[0], 1, 1
                zoneType = 1
            elif len(shape) == 2:
                shape = 1, shape[0], shape[1]
            d["varTypes"] = [self.get_varTypes(
                name_zone, j) for j in self.nameVars]
            d["passiveVarList"] = [
                0 if zone_value.get(i, empty).size > 0 else 1 for i in self.nameVars
            ]
            d["shareVarFromZone"] = [0] * len(self.nameVars)
            # valueLocation: value 1 represent the data is saved on nodes, value 0 means on elements center
            d["valueLocation"] = value_location
            d["IJK"] = shape
            d["zoneType"] = zoneType
            d["solutionTime"] = 0.0
            d["strandID"] = 0
            d["shareConnectivityFromZone"] = 0
            d["faceNeighborMode"] = 0
            d["numFaceConnections"] = 0
            d["parentZone"] = 0
            d["name"] = name_zone
            return d

        temp_zone = self[self.nameZones[0]]
        value_location = [sum(temp_zone[key].shape) for key in self.nameVars]
        max_location = max(value_location)
        value_location = [0 if i < max_location else 1 for i in value_location]
        self.zone_info = [cal_zone_info(i, value_location)
                          for i in self.nameZones]
        self.fileType = 0
        self.added_new_zone = False

    def get_varTypes(self, name_zone, name_var):
        varTypes = {"int32": 3, "float64": 2, "float32": 1}
        d = self[name_zone][name_var]
        dtype = str(d.dtype)
        if dtype == "int64":
            d = d.astype("int32")
            self[name_zone][name_var] = d
            dtype = "int32"
        return varTypes[dtype]


class write_tecio:
    fileFormat = 0  # .szplt

    def __init__(self, filename, dataset=None, verbose=True):
        """
        dataset 只要是包含两层字典的数据都可以 like d[key_zone][key_var],如果是非SzpltData类型的数据，目前只支持结构化的数据
        """
        self.filename = filename
        self.verbose = verbose
        if hasattr(dataset, "added_new_zone") and dataset.added_new_zone:
            dataset = {
                k: {k2: dataset[k][k2] for k2 in dataset[k].keys()}
                for k in dataset.keys()
            }
        if not isinstance(dataset, SzpltData):
            dataset = SzpltData(dataset)
        self.dataset = dataset
        self.dll = GLOBAL_DLL
        self.filehandle = self._get_filehandle()
        empty = np.array([])
        for i, zone_name in enumerate(dataset.nameZones):
            info = dataset.zone_info[i]
            I, J, K = info["IJK"]
            zone_set = dataset[zone_name]
            varTypes = self._list_to_int_array(info["varTypes"])

            shareVarFromZone = self._list_to_int_array(
                info["shareVarFromZone"])
            valueLocation = self._list_to_int_array(info["valueLocation"])
            info["passiveVarList"] = [
                0 if np.size(zone_set[i]) > 0 else 1 for i in dataset.nameVars
            ]
            passiveVarList = self._list_to_int_array(info["passiveVarList"])

            if info["zoneType"] == Structed_Grid:
                outputZone = self._tecZoneCreateIJK(
                    zone_name,
                    I,
                    J,
                    K,
                    varTypes,
                    shareVarFromZone,
                    valueLocation,
                    passiveVarList,
                    info["shareConnectivityFromZone"],
                    info["numFaceConnections"],
                    info["faceNeighborMode"],
                )
            else:
                outputZone = self._tecZoneCreateFE(
                    zone_name,
                    info["zoneType"],
                    I,
                    J,
                    varTypes,
                    shareVarFromZone,
                    valueLocation,
                    passiveVarList,
                    info["shareConnectivityFromZone"],
                    info["numFaceConnections"],
                    info["faceNeighborMode"],
                )
            self._tecZoneSetUnsteadyOptions(
                outputZone, info["solutionTime"], info["strandID"]
            )
            if info["parentZone"] != 0:
                self._tecZoneSetParentZone(outputZone, info["parentZone"])
            if info["numFaceConnections"] > 0:
                faceConnections = info["faceConnections"]
                if isinstance(faceConnections, list) or isinstance(
                    faceConnections, tuple
                ):
                    faceConnections = np.array(faceConnections, dtype="int64")
                    print(faceConnections)
                if faceConnections.itemsize == 8:
                    self._write_data_all_type(
                        self.dll.tecZoneFaceNbrWriteConnections64,
                        faceConnections.ctypes,
                        outputZone,
                    )
                else:
                    self._write_data_all_type(
                        self.dll.tecZoneFaceNbrWriteConnections32,
                        faceConnections.ctypes,
                        outputZone,
                    )
            if info.get("aux") is not None:
                for key, value in info["aux"].items():
                    key_p = ctypes.c_char_p(key.encode())
                    value_p = ctypes.c_char_p(value.encode())
                    self.dll.tecZoneAddAuxData(
                        self.filehandle, outputZone, key_p, value_p
                    )
            for j, var_name in enumerate(dataset.nameVars):

                var_n = j + 1
                data = zone_set[var_name]  # .copy(order="C")
                if info["zoneType"] is Structed_Grid:
                    # if data.ndim == 1:
                    # data.shape = dataset.zone_info[i]['IJK']
                    # if data.ndim == 2:
                    # shape = data.shape
                    # data.shape = 1, shape[0], shape[1]
                    if data.size > 0:
                        data.shape = dataset.zone_info[i]["IJK"]
                        data = np.reshape(data, data.shape, order="F")
                        # print(data.shape)
                        # data = data.transpose((2, 1, 0)).copy()
                ff = [min(i, j) for j in info["shareVarFromZone"]]
                if info["passiveVarList"][var_n - 1] == 0 and ff[var_n - 1] == 0:

                    fieldDataType = info["varTypes"][var_n - 1]
                    if fieldDataType is FieldDataType_Float:
                        self._write_data_all_type(
                            self.dll.tecZoneVarWriteFloatValues,
                            data.ctypes,
                            outputZone,
                            var_n,
                            0,
                            data.size,
                        )
                    elif fieldDataType is FieldDataType_Double:
                        self._write_data_all_type(
                            self.dll.tecZoneVarWriteDoubleValues,
                            data.ctypes,
                            outputZone,
                            var_n,
                            0,
                            data.size,
                        )
                    elif fieldDataType is FieldDataType_Int32:
                        self._write_data_all_type(
                            self.dll.tecZoneVarWriteInt32Values,
                            data.ctypes,
                            outputZone,
                            var_n,
                            0,
                            data.size,
                        )
                    elif fieldDataType is FieldDataType_Int16:
                        self._write_data_all_type(
                            self.dll.tecZoneVarWriteInt16Values,
                            data.ctypes,
                            outputZone,
                            var_n,
                            0,
                            data.size,
                        )
                    elif fieldDataType is FieldDataType_Byte:
                        self._write_data_all_type(
                            self.dll.tecZoneVarWriteByteValues,
                            data.ctypes,
                            outputZone,
                            var_n,
                            0,
                            data.size,
                        )
                    else:
                        print(fieldDataType, "iiiiiiiiiiiii")
                        raise Exception(
                            "FieldDataType Error:not defined data type")
            self._write_zone_node_map(outputZone, info, zone_set)

    def _write_zone_node_map(self, zone_n, info, zone_set):
        # info = self.dataset.zone_info[self.dataset.nameZones[zone_n-1]]
        if (
            info["zoneType"] is not Structed_Grid
            and info["shareConnectivityFromZone"] == 0
        ):
            Elements = zone_set.Elements
            numValues = Elements.size
            if Elements.itemsize == 8:
                # is64bit True
                self._write_data_all_type(
                    self.dll.tecZoneNodeMapWrite64,
                    Elements.ctypes,
                    zone_n,
                    0,
                    1,
                    numValues,
                )
            else:
                self._write_data_all_type(
                    self.dll.tecZoneNodeMapWrite32,
                    Elements.ctypes,
                    zone_n,
                    0,
                    1,
                    numValues,
                )

    def _list_to_int_array(self, l):
        t = (ctypes.c_int * len(l))()

        for i, j in enumerate(l):
            t[i] = j
        return t

    def _get_filehandle(self):
        p = ctypes.c_int(13)
        p1 = ctypes.pointer(p)
        filehandle = ctypes.pointer(p1)
        name = ctypes.c_char_p(self.filename.encode())
        fileType = self.dataset.fileType
        name_str = ",".join([str(i) for i in self.dataset.nameVars])
        # name_str
        var_list_str = ctypes.c_char_p(name_str.encode())
        title_str = ctypes.c_char_p(self.dataset.title.encode())
        if self.filename.endswith(".szplt"):
            fileFormat = 1
        else:
            raise Exception("file format error")
        self.dll.tecFileWriterOpen(
            name, title_str, var_list_str, fileFormat, fileType, 2, None, filehandle
        )

        if self.verbose:
            outputDebugInfo = 1
            self.dll.tecFileSetDiagnosticsLevel(filehandle[0], outputDebugInfo)
        return filehandle[0]

    def _tecZoneCreateIJK(
        self,
        zoneTitle,
        iMax,
        jMax,
        kMax,
        varTypes,
        shareVarFromZone,
        valueLocation,
        passiveVarList,
        shareConnectivityFromZone,
        numFaceConnections,
        faceNeighborMode,
    ):
        p = ctypes.pointer(ctypes.c_int(0))
        zone_title = ctypes.c_char_p(zoneTitle.encode())
        self.dll.tecZoneCreateIJK(
            self.filehandle,
            zone_title,
            iMax,
            jMax,
            kMax,
            varTypes,
            shareVarFromZone,
            valueLocation,
            passiveVarList,
            shareConnectivityFromZone,
            numFaceConnections,
            faceNeighborMode,
            p,
        )
        return p[0]

    def _tecZoneCreateFE(
        self,
        zoneTitle,
        zoneType,
        iMax,
        jMax,
        varTypes,
        shareVarFromZone,
        valueLocation,
        passiveVarList,
        shareConnectivityFromZone,
        numFaceConnections,
        faceNeighborMode,
    ):
        t = ctypes.c_int(0)
        p = ctypes.pointer(t)
        zone_title = ctypes.c_char_p(zoneTitle.encode())

        self.dll.tecZoneCreateFE(
            self.filehandle,
            zone_title,
            zoneType,
            iMax,
            jMax,
            varTypes,
            shareVarFromZone,
            valueLocation,
            passiveVarList,
            shareConnectivityFromZone,
            numFaceConnections,
            faceNeighborMode,
            p,
        )
        return p[0]

    def _tecZoneSetUnsteadyOptions(self, zone_n, solutionTime=0, StrandID=0):
        if solutionTime != 0 or StrandID != 0:
            solutionTime = ctypes.c_double(solutionTime)
            self.dll.tecZoneSetUnsteadyOptions(
                self.filehandle, zone_n, solutionTime, StrandID
            )

    def _tecZoneSetParentZone(self, zone_n, zone_parent):
        self.dll.tecZoneSetParentZone(self.filehandle, zone_n, zone_parent)

    def _write_data_all_type(self, fun, data, *d):
        fun(self.filehandle, *d, data)

    def close(self):
        self.dll.tecFileWriterClose(ctypes.pointer(self.filehandle))


def read(filename, isread=False):
    return SzpltData(filename, isread)


def write(filename, dataset, verbose=True):
    t = write_tecio(filename, dataset, verbose=verbose)
    t.close()


def cal_zone(number, g, q):
    g = g[number]
    q = q[number]
    k = {i: g[i] for i in "XYZ"}
    y = {"VAR{}".format(key): val for key, val in q.items()
         if isinstance(key, int)}
    k.update(y)
    return k


# % functions for INCA
def ReadSinglePlt(file_nm, var_list=None):
    file = SzpltData(filename=file_nm)
    zone_num = file.numZones
    SolTime = file.zone_info[0]["solutionTime"]
    df = pd.DataFrame()
    for i in range(zone_num):
        zonename = file.nameZones[i]
        ZoneDict = file[zonename]
        temp = pd.DataFrame(ZoneDict)
        df = pd.concat([df, temp])
    if var_list is None:
        df_select = df
    else:
        df_select = df[var_list]
    file.close()
    return (df_select, SolTime)


def ReadMultiPlt(folder, var_list=None):
    FileList = os.listdir(folder)
    df, SolTime = ReadSinglePlt(folder + FileList[0], var_list)
    for i in range(np.size(FileList) - 1):
        df1, _ = ReadSinglePlt(folder + FileList[i + 1], var_list)
        df = pd.concat([df, df1])
    return (df, SolTime)


def ReadINCAResults(
    FoldPath,
    VarList=None,
    SubZone=None,
    FileName=None,
    SpanAve=None,
    SavePath=None,
    OutFile=None,
    opt=1,
):
    
    if FileName is None:
        dirs = glob(FoldPath + '*plt')
        FileName = sorted(dirs)
        # files = sorted(os.listdir(FoldPath))
        # FileName = [os.path.join(FoldPath, name) for name in files]
    if isinstance(FileName, list):
        szplt = FileName[0].find("szplt")
        if szplt != -1:
            df, SolTime = ReadSinglePlt(FileName[0], VarList)
            for i in range(np.size(FileName) - 1):
                df1, _ = ReadSinglePlt(FileName[i + 1], VarList)
                df = pd.concat([df, df1])
        else:
            print("There is no szplt files!!!")
    else:
        szplt = FileName.find("szplt")
        if szplt != -1:
            df, SolTime = ReadSinglePlt(FileName, VarList)
        else:
            print("There is no szplt files!!!")
    # if Equ is not None:
    #     for i in range(np.size(Equ)):
    #        tp.data.operate.execute_equation(Equ[i])

    # df = df.drop_duplicates(keep='last')  # if on,spanwise-average may wrong
    if SpanAve is True:
        grouped = df.groupby(["x", "y"])
        df = grouped.mean().reset_index()
    if SpanAve is None:
        grouped = df.groupby(["x", "y", "z"])
        df = grouped.mean().reset_index()
    if opt == 2:
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["z"] = df["z"].astype(float)
        if SubZone is not None:
            df = df.query(
                "x>={0} & x<={1} & y<={2}".format(
                    SubZone[0][0], SubZone[0][1], SubZone[1][1]
                )
            )
    if SavePath is not None and OutFile is not None:
        st = "%08.2f" % SolTime
        df.to_hdf(SavePath + OutFile + "_" + st + ".h5", "w", format="fixed")
    return (df, SolTime)


def create_fluc_hdf(path, path_m, outpath, SpanAve=None):
    df1, st = ReadINCAResults(path, SpanAve=SpanAve)
    df2, _ = ReadINCAResults(path_m, SpanAve=SpanAve)
    df = df1
    df["u`"] = df1["u"] - df2["<u>"]
    df["v`"] = df1["v"] - df2["<v>"]
    df["w`"] = df1["w"] - df2["<w>"]
    df["rho`"] = df1["rho"] - df2["<rho>"]
    df["p`"] = df1["p"] - df2["<p>"]
    df["T`"] = df1["T"] - df2["<T>"]
    if SpanAve is True:
        grouped = df.groupby(["x", "y"])
        df = grouped.mean().reset_index()
    if SpanAve is None:
        grouped = df.groupby(["x", "y", "z"])
        df = grouped.mean().reset_index()
    st = "%08.2f" % st
    df.to_hdf(outpath + "TP_data_" + st + ".h5", "w", format="fixed")


def create_fluc_tec(file, file_m, outpath, var, var1, var2):
    ds1 = SzpltData(file)
    ds2 = SzpltData(file_m)
    # var = ['u`', 'v`', 'w`', 'rho`', 'p`', 'T`']
    # var1 = ['u', 'v', 'w', 'rho', 'p', 'T']
    # var2 = ['<u>', '<v>', '<w>', '<rho>', '<p>', '<T>']
    nm = [i[-4:] for i in ds1.nameZones]
    nm_m = [i[-4:] for i in ds2.nameZones]
    for i in range(ds1.numZones):
        z_nm = ds1.nameZones[i]
        ind = nm_m.index(nm[i])
        z_nm_m = ds2.nameZones[ind]
        for j in range(np.size(var1)):
            ds1[z_nm][var[j]] = ds1[z_nm][var1[j]]-ds2[z_nm_m][var2[j]]
    t = write_tecio(outpath + os.path.basename(file), ds1, verbose=True)
    t.close()


def span_ave_tec(files, outpath):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for j in range(np.size(files)):
        ds = SzpltData(files[j])
        for i in range(ds.numZones):
            z_nm = ds.nameZones[i]
            df_dict = ds[z_nm]
            df = pd.DataFrame.from_dict(df_dict)
            grouped = df.groupby(["x", "y"])
            df_ave = grouped.mean().reset_index()
            df_ave = df_ave.sort_values(by=['y', 'x'])

            for ky in ds[z_nm].keys():
                ds[z_nm][ky] = np.asarray(df_ave[ky], dtype=np.float32)
            dim = ds.zone_info[i]['IJK']
            dim = (dim[0], dim[1], 1)
            ds.zone_info[i]['IJK'] = dim
        ds.close()
        t = write_tecio(outpath + os.path.basename(files[j]), ds, verbose=True)
        t.close()



def create_fluc_inca(path, path_m, outpath):
    files = sorted(os.listdir(path))
    file_nm = [os.path.join(path, name) for name in files]
    files_m = sorted(os.listdir(path_m))
    file_m_nm = [os.path.join(path_m, name) for name in files_m]

    szplt = [x for x in file_nm if x.endswith(".szplt")]
    szplt_m = [x for x in file_m_nm if x.endswith(".szplt")]

    var = ['u`', 'v`', 'w`', 'rho`', 'p`', 'T`']
    var1 = ['u', 'v', 'w', 'rho', 'p', 'T']
    var2 = ['<u>', '<v>', '<w>', '<rho>', '<p>', '<T>']

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    # if np.size(szplt) == 1:
    if np.size(szplt) == np.size(szplt_m):
        for i in range(np.size(szplt)):
            file1 = szplt[i]
            file2 = szplt_m[i]
            create_fluc_tec(file1, file2, outpath,
                            var=var, var1=var1, var2=var2)

    elif np.size(szplt) < np.size(szplt_m):
        file1 = szplt[0]
        ds1 = SzpltData(file1)
        nm1 = [i[-4:] for i in ds1.nameZones]
        file3 = 'TP_fluc_'
        for i in range(np.size(szplt_m)):
            file2 = szplt_m[i]
            ds2 = SzpltData(file2)
            z_nm2 = ds2.nameZones[0]
            ind = nm1.index(z_nm2[-4:])
            z_nm1 = ds1.nameZones[ind]
            for j in range(np.size(var1)):
                # ind = list(ds1.nameVars_dict.keys()).index(var1[j])
                # val = ds1._read_zone_var(i+1, ind+1)
                ds1[z_nm1][var[j]] = ds1[z_nm1][var1[j]] - ds2[z_nm2][var2[j]]
            ds2.close()

        t = write_tecio(outpath + file3 + z_nm2[-4:] + '.szplt',
                        ds1, verbose=True)
        t.close()

    else:
        file2 = szplt_m[0]
        ds2 = SzpltData(file2)
        nm2 = [i[-4:] for i in ds2.nameZones]
        file3 = 'TP_fluc_'
        for i in range(np.size(szplt)):
            file1 = szplt[i]
            ds1 = SzpltData(file1)
            z_nm1 = ds1.nameZones[0]
            ind = nm2.index(z_nm1[-4:])
            z_nm2 = ds2.nameZones[ind]
            for j in range(np.size(var2)):
                # ind = list(ds2.nameVars_dict.keys()).index(var2[j])
                # val = ds2._read_zone_var(i+1, ind+1)
                ds1[z_nm1][var[j]] = ds1[z_nm1][var1[j]] - ds2[z_nm2][var2[j]]
            ds1.close()
            t = write_tecio(outpath + file3 + z_nm1[-4:] + '.szplt',
                            ds1, verbose=True)
            t.close()
            

# %%
if __name__ == "__main__":
    # path = "/mnt/work/cases/wavy_1019/TP_data_00093047/"
    # finm = path + "TP_dat_000001.szplt"
    # var_list = ["x", "y", "z", "u"]
    # df, s_time = ReadSinglePlt(finm, var_list)
    # df, s_time = ReadMultiPlt(path, var_list)
    # df, s_time = ReadINCAResults(path, var_list, SpanAve=True)
    # df.to_hdf(path + 'TP_data_.h5', 'w', format='fixed')
    # file = pt.SzpltData(filename=path + 'TP_dat_000002.szplt')
    # t = pt.write_tecio(path+'test.szplt', file, verbose=True)
    # create_fluc_inca(
    #     "E:/cases/wavy_0918/snapshot_00100056/",
    #     "E:/cases/wavy_0918/TP_stat_ave/",
    #     "E:/cases/wavy_0918/TP_fluc_00100056/",
    # )
    # path = '/mnt/work/cases_new/heating2/'
    path = '/media/weibo/Weibo_data/2023cases/flat/'
    path1 = path + 'TP_stat/'
    ReadINCAResults(path1, SpanAve=True, SavePath=path, OutFile='TP_data')


    # %%
    # pathSN = path + 'snapshots/'
    # slicenm = '/TP_2D_S_011.szplt'
    # col = ["x", "y", "z", "u", 'v', 'w', 'p', 'rho', 'T']
    # dirs = sorted(os.listdir(pathSN))
    # print("loading data")
    # DataFrame, _ = ReadSinglePlt(pathSN + dirs[0] + slicenm)
    # grouped = DataFrame.groupby(['x', 'y', 'z'])
    # DataFrame = grouped.mean().reset_index()
    # with timer("Load Data"):
    #     for i in range(np.size(dirs)-1):
    #         TempFrame, _ = ReadSinglePlt(pathSN + dirs[i + 1] + slicenm)
    #         grouped = TempFrame.groupby(['x', 'y', 'z'])
    #         TempFrame = grouped.mean().reset_index()
    #         if np.shape(TempFrame)[0] != np.shape(DataFrame)[0]:
    #             sys.exit('The input snapshots does not match!!!')
    #         DataFrame += TempFrame

    # # obtain mean flow 
    # meanflow = DataFrame/np.size(dirs)
    # print("data loaded")
    # meanflow.to_hdf(path + "MeanFlow1.h5", "w", format="fixed")

    """
    path_m = '/mnt/share/cases/base/TP_stat/'
    files_m = sorted(os.listdir(path_m))
    file_m_nm = [os.path.join(path_m, name) for name in files_m]
    szplt = [x for x in file_m_nm if x.endswith(".szplt")]
    span_ave_tec(szplt, '/mnt/share/cases/base/TP_stat_ave/')
    
    create_fluc_inca(
        "/mnt/share/cases/base/snapshot_00000297/",
        "/mnt/share/cases/base/TP_stat_ave/",
        "/mnt/share/cases/base/TP_fluc_00000297/",
    )
    """

    # for i in range(np.size(szplt)):
    #     file1 = szplt[i]
    #    span_ave_tec(file1, 'E:/cases/wavy_0918/TP_stat_ave/')
    """
    path = 'E:/cases/wavy_0918/snapshot_00100056/'
    path_m = 'E:/cases/wavy_0918/TP_stat/'
    outpath = 'E:/cases/wavy_0918/TP_stat_ave/'

    file = '/mnt/share/cases/wavy_0918/TP_stat/TP_stat_000001.szplt'
    file = 'E:/cases/wavy_0918/TP_stat/TP_stat_000001.szplt'
    file1 = 'E:/cases/wavy_0918/TP_stat/test.szplt'
    ds = SzpltData(file)
    ds.numZones
    z_nm = ds.nameZones[0]
    df_dict = ds[z_nm]
    df = pd.DataFrame.from_dict(df_dict)
    grouped = df.groupby(["x", "y"])
    df_ave = grouped.mean().reset_index()
    df_ave = df_ave.sort_values(by=['y', 'x'])
    df_ave_dict = df_ave.to_dict('list')

    for ky in ds[z_nm].keys():
        ds[z_nm][ky] = np.asarray(df_ave[ky], dtype=np.float32)
    dim = ds.zone_info[0]['IJK']
    dim = (dim[0], dim[1], 1)
    ds.zone_info[0]['IJK'] = dim
    t = write_tecio(file1, ds, verbose=True)
    t.close()
    """

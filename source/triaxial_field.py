# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 3D data
@author: Weibo Hu
"""

import pandas as pd
from planar_field import PlanarField
import plt2pandas as p2p
from glob import glob
import numpy as np


class TriField(PlanarField):
    def __init__(self):
        super().__init__()

    @property
    def TriData(self):
        return self._data_field

    @TriData.setter
    def TriData(self, frame):
        assert isinstance(frame, pd.DataFrame)
        self._TriData = frame

    @property
    def PlanarData(self):
        return self._PlanarData

    @PlanarData.setter
    def PlanarData(self, frame):
        assert isinstance(frame, pd.DataFrame)
        self._PlanarData = frame

    def load_3data(self, path, FileList=None, NameList=None):
        # nfiles = np.size(os.listdir(path))
        if FileList is None:
            infile = glob(path + '*.plt')
        else:
            infile = FileList

        if NameList is None:
            df = p2p.ReadAllINCAResults(path,
                                        FileName=infile,
                                        SpanAve=None)
        elif NameList == 'h5':
            if np.size(FileList) == 1:
                if isinstance(infile, list):
                    df = pd.read_hdf(infile[0])
                else:
                    df = pd.read_hdf(infile)
            else:
                num = np.size(FileList)
                df = pd.concat([pd.read_hdf(FileList[i]) for i in range(num)])
                df.reset_index()
        else:
            df = p2p.ReadINCAResults(path,
                                     VarList=NameList,
                                     FileName=infile,
                                     SpanAve=False)

        # df = df.drop_duplicates(keep='last')
        self._data_field = df

    def span_ave(self):
        grouped = self._data_field.groupby(['x', 'y'])
        df = grouped.mean().reset_index()
        return (df)

    def unique(self):
        grouped = self._data_field.groupby(['x', 'y', 'z'])
        df = grouped.mean().reset_index()
        self._data_field = df
        return (df)

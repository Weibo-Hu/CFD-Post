# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 3D data 
@author: Weibo Hu
"""

import pandas as pd
from line_field import LineField
from planar_field import PlanarField
from timer import timer
import os
import plt2pandas as p2p
from glob import glob
import numpy as np


class TriField(PlanarField):
    def __init__(self):
        super().__init__()
    
    @property
    def TriData(self):
        return self._data_field
    
    @property
    def PlanarData(self):
        grouped = self._data_field.groupby(['x', 'y', 'z'])
        df = grouped.mean().reset_index()
        self._PlanarData = df
        return self._PlanarData

    def load_3data(self, path, FileList=None, NameList=None):
        nfiles = np.size(os.listdir(path))
        if FileList is None:
            infile = glob(path + '*.plt')
        else:
            infile = FileList

        if NameList is None:
            df = p2p.ReadAllINCAResults(nfiles,
                                        path,
                                        FileName=infile,
                                        SpanAve=False)
        else:
            df = p2p.ReadINCAResults(nfiles,
                                     path,
                                     VarList=NameList,
                                     FileName=infile,
                                     SpanAve=False)   

        df = df.drop_duplicates(keep='last')    
        self._data_field = df




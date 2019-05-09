# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 2D data
@author: Weibo Hu
"""

import pandas as pd
from line_field import LineField
from timer import timer
import os
import plt2pandas as p2p


class PlanarField(LineField):
    def __init__(self):
        super().__init__()
        # self.PlanarData = pd.DataFrame()

    @property
    def PlanarData(self):
        return self._data_field

    @property
    def u_m(self):
        return self._data_field['<u>'].values

    @property
    def v_m(self):
        return self._data_field['<v>'].values

    @property
    def w_m(self):
        return self._data_field['<w>'].values

    @property
    def rho_m(self):
        return self._data_field['<rho>'].values

    @property
    def R11(self):
        return self._data_field['<u`u`>'].values

    @property
    def R22(self):
        return self._data_field['<v`v`>'].values

    @property
    def R33(self):
        return self._data_field['<w`w`>'].values

    @property
    def R12(self):
        return self._data_field['<u`v`>'].values

    @property
    def R13(self):
        return self._data_field['<u`w`>'].values

    @property
    def R23(self):
        return self._data_field['<v`w`>'].values

    def load_meanflow(self, path, nfiles=49):
        exists = os.path.isfile(path + 'MeanFlow/MeanFlow.h5')
        if exists:
            self._data_field = pd.read_hdf(path + 'MeanFlow/MeanFlow.h5')
        else:
            with timer('load mean flow from tecplot data'):
                df = p2p.ReadAllINCAResults(nfiles,
                                            path + 'TP_stat/',
                                            path + 'MeanFlow/',
                                            SpanAve=True,
                                            OutFile='MeanFlow')
            self._data_field = df

    def yprofile(self, var_name, var_val):
        df1 = self.PlanarData.loc[self.PlanarData[var_name] == var_val]
        df2 = PlanarField.uniq1d(df1, 'y')
        df3 = df2.sort_values(by=['y'], ascending=True)
        return (df3)

    def add_walldist(self, StepHeight):
        self.PlanarData['walldist'] = self.PlanarData['y']
        self.PlanarData.loc[self.PlanarData['x'] > 0.0, 'walldist'] \
            += StepHeight

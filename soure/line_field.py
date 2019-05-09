# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 1D data
@author: Weibo Hu
"""

import pandas as pd
import sys


class LineField(object):
    def __init__(self):
        """ """
        self._data_field = pd.DataFrame()
        self.ProbeSignal = pd.DataFrame()

    @property
    def ProbeSignal(self):
        return self._data_field

    @ProbeSignal.setter
    def ProbeSignal(self, frame):
        assert isinstance(frame, pd.DataFrame)
        self._data_field = frame

    @property
    def x(self):
        return self._data_field['x'].values

    @property
    def y(self):
        return self._data_field['y'].values

    @property
    def z(self):
        return self._data_field['z'].values

    @property
    def time(self):
        return self._data_field['Solution Time'].values

    @property
    def walldist(self):
        return self._data_field['walldist'].values

    @property
    def u(self):
        return self._data_field['u'].values

    @property
    def v(self):
        return self._data_field['v'].values

    @property
    def w(self):
        return self._data_field['w'].values

    @property
    def rho(self):
        return self._data_field['rho'].values

    @property
    def p(self):
        return self._data_field['p'].values

    @property
    def Mach(self):
        return self._data_field['Mach'].values

    @property
    def T(self):
        return self._data_field['T'].values

    @property
    def mu(self):
        return self._data_field['mu'].values

    def add_variable(self, var_name, var_val):
        if var_name in self._data_field.columns:
            sys.exit(var_name + "is already in dataframe")
        else:
            self._data_field[var_name] = var_val

    @classmethod
    def uniq1d(cls, frame, var_name):
        grouped = frame.groupby([var_name])
        df = grouped.mean().reset_index()
        return (df)


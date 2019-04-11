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
        self.ProbeSignal = pd.DataFrame()

    @property
    def ProbeSignal(self):
        return self._ProbeSignal

    @ProbeSignal.setter
    def ProbeSignal(self, frame):
        assert isinstance(frame, pd.DataFrame)
        self._ProbeSignal = frame

    @property
    def x(self):
        return self.ProbeSignal['x'].values

    @property
    def y(self):
        return self.ProbeSignal['y'].values

    @property
    def z(self):
        return self.ProbeSignal['z'].values

    @property
    def time(self):
        return self.ProbeSignal['Solution Time'].values

    @property
    def walldist(self):
        return self.ProbeSignal['walldist'].values

    @property
    def u(self):
        return self.ProbeSignal['u'].values

    @property
    def v(self):
        return self.ProbeSignal['v'].values

    @property
    def w(self):
        return self.ProbeSignal['w'].values

    @property
    def rho(self):
        return self.ProbeSignal['rho'].values

    @property
    def p(self):
        return self.ProbeSignal['p'].values

    @property
    def Mach(self):
        return self.ProbeSignal['Mach'].values

    @property
    def T(self):
        return self.ProbeSignal['T'].values

    @property
    def mu(self):
        return self.ProbeSignal['mu'].values

    def add_variable(self, var_name, var_val):
        if var_name in self.ProbeSignal.columns:
            sys.exit(var_name + "is already in dataframe")
        else:
            self.ProbeSignal[var_name] = var_val

    @classmethod
    def uniq1d(cls, frame, var_name):
        grouped = frame.groupby([var_name])
        df = grouped.mean().reset_index()
        return (df)


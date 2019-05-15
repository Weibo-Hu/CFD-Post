# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 1D data
@author: Weibo Hu
"""

import pandas as pd
import sys
import numpy as np


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
        if 'Solution Time' in self._data_field.columns:
            return self._data_field['Solution Time'].values
        if 'time' in self._data_field.columns:
            return self._data_field['time'].values

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
    
    @property
    def u_m(self):
        return self._data_field['u'].values.mean()
    
    @property
    def v_m(self):
        return self._data_field['v'].values.mean()
    
    @property
    def w_m(self):
        return self._data_field['w'].values.mean()
    
    @property
    def rho_m(self):
        return self._data_field['rho'].values.mean()
    
    @property
    def T_m(self):
        return self._data_field['T'].values.mean()
    
    @property
    def p_m(self):
        return self._data_field['p'].values.mean()

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

    def probe_file(self, path, loc):
        infile = open(path + 'inca_probes.inp')
        probe = np.loadtxt(infile, skiprows=6, usecols=(1,2,3,4))
        infile.close()
        xarr = np.around(probe[:,1], 3)
        yarr = np.around(probe[:,2], 3)
        zarr = np.around(probe[:,3], 3)
        probe_ind = np.where( (xarr[:]==np.around(loc[0], 3))
                            & (yarr[:]==np.around(loc[1], 3))
                            & (zarr[:]==np.around(loc[2], 3)) )
        if len(probe_ind[0]) == 0:
            print("The probe you input is not found in the probe files!!!")
        probe_num = probe_ind[0][-1] + 1
        filename = 'probe_' + format(probe_num, '05d') + '.dat'
        return(filename)

    def load_probe(self, path, loc, per=None, varname=None, uniq=None):
        if varname == None:
            varname = ['itstep', 'time', 'u', 'v', 'w',
                       'rho', 'E', 'p']
        filename = self.probe_file(path, loc)
        print('Probe file name:', filename)
        data = pd.read_csv(path + filename, sep=' ',
                           index_col=False, header=None, names=varname,
                           skiprows=2, skipinitialspace=True)
        if per is not None:
            ind = data['time'].between(per[0], per[1], inclusive=True)
            data = data[ind]
        if uniq == None:
            data = data.drop_duplicates(keep='last')
        data = data.sort_values(by=['time'])
        self._data_field = data
        return(data)

    def extract_data(self, per):
        ind = self._data_field['time'].between(per[0], per[1],
                                                inclusive=True)
        data = self._data_field[ind]
        self._data_field = data
        return(data)




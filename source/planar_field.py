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
import pytecio as pytec
from glob import glob
import numpy as np


class PlanarField(LineField):
    def __init__(self):
        super().__init__()
        self._PlanarData = pd.DataFrame()

    @property
    def PlanarData(self):
        self._PlanarData = self._data_field
        # return self._data_field
        return self._PlanarData

    @PlanarData.setter
    def PlanarData(self, df):
        self._PlanarData = df

    @property
    def u_m(self):
        return(self._data_field['<u>'].values)

    @property
    def v_m(self):
        return(self._data_field['<v>'].values)

    @property
    def w_m(self):
        return(self._data_field['<w>'].values)

    @property
    def rho_m(self):
        return(self._data_field['<rho>'].values)

    @property
    def p_m(self):
        return(self._data_field['<p>'].values)

    @property
    def T_m(self):
        return(self._data_field['<T>'].values)

    @property
    def mu_m(self):
        return(self._data_field['<mu>'].values)

    @property
    def vorticity_1_m(self):
        return(self._data_field['<vorticity_1>'].values)

    @property
    def vorticity_2_m(self):
        return(self._data_field['<vorticity_2>'].values)

    @property
    def vorticity_3_m(self):
        return(self._data_field['<vorticity_3>'].values)

    @property
    def R11(self):
        return(self._data_field['<u`u`>'].values)

    @property
    def R22(self):
        return(self._data_field['<v`v`>'].values)

    @property
    def R33(self):
        return(self._data_field['<w`w`>'].values)

    @property
    def R12(self):
        return(self._data_field['<u`v`>'].values)

    @property
    def R13(self):
        return(self._data_field['<u`w`>'].values)

    @property
    def R23(self):
        return(self._data_field['<v`w`>'].values)

    def load_data(self, path, FileList=None, ExtName=None):
        # nfiles = np.size(os.listdir(path))
        if FileList is None:
            infile = glob(path + '*plt')
        else:
            infile = FileList

        if ExtName is None:
            # ext_name = os.path.splitext(infile)
            df = p2p.ReadAllINCAResults(path,
                                        FileName=infile)
        elif ExtName == 'h5':
            df = pd.read_hdf(infile)
        elif ExtName == 'tecio':
            df, SolTime = pytec.ReadSinglePlt(infile)
        else:
            df = p2p.ReadINCAResults(path,
                                     VarList=ExtName,
                                     FileName=infile)
        df = df.drop_duplicates(keep='last')
        grouped = df.groupby(['x', 'y', 'z'])
        df = grouped.mean().reset_index()
        self._data_field = df

    def load_meanflow(self, path, FileList=None, lib='tecio'):
        exists = os.path.isfile(path + 'MeanFlow/MeanFlow.h5')
        if exists:
            df = pd.read_hdf(path + 'MeanFlow/MeanFlow.h5')
            df = df.drop_duplicates(keep='last')
            grouped = df.groupby(['x', 'y', 'z'])
            df = grouped.mean().reset_index()
        else:
            equ = ['{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)']
            # nfiles = np.size(os.listdir(path + 'TP_stat/'))
            print('try to calculate data')
            with timer('load mean flow from tecplot data'):
                if FileList is None:
                    if lib == 'tecio':
                        df, SolTime = pytec.ReadINCAResults(path + 'TP_stat/',
                                                            SavePath=path + 'MeanFlow/',
                                                            SpanAve=True,
                                                            OutFile='MeanFlow')

                    else:
                        df = p2p.ReadAllINCAResults(path + 'TP_stat/',
                                                    path + 'MeanFlow/',
                                                    SpanAve=True,
                                                    Equ=equ,
                                                    OutFile='MeanFlow')
                else:
                    if lib == 'tecio':
                        df, SolTime = pytec.ReadINCAResults(path + 'TP_stat/',
                                                            SavePath=path + 'MeanFlow/',
                                                            FileName=FileList,
                                                            SpanAve=True,
                                                            OutFile='MeanFlow')
                    else:
                        df = p2p.ReadAllINCAResults(path + 'TP_stat/',
                                                    path + 'MeanFlow/',
                                                    FileName=FileList,
                                                    SpanAve=True,
                                                    Equ=equ,
                                                    OutFile='MeanFlow')

            print('done with saving data')
        self._data_field = df


    def merge_stat(self, path, filenm='MeanFlow*'):
        # dirs = sorted(os.listdir(path))
        dirs = glob(path + filenm)
        for i in np.arange(np.size(dirs)):
            if i == 0:
                flow = pd.read_hdf(dirs[i])
            else:
                df = pd.read_hdf(dirs[i])
                flow = pd.concat([flow, df])
        flow = flow.drop_duplicates(keep='last')
        # grouped = flow.groupby(['x', 'y', 'z'])
        flow.to_hdf(path + 'MeanFlow.h5', 'w', format='fixed')
        self._data_field = flow


    def merge_meanflow(self, path):
        dirs = sorted(os.listdir(path + 'TP_stat/'))
        nfiles = np.size(dirs)
        equ = ['{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)']
        for i in np.arange(nfiles):
            FileList = os.path.join(path + 'TP_stat/', dirs[i])
            with timer(FileList):
                df = p2p.ReadAllINCAResults(path + 'TP_stat/',
                                            path + 'MeanFlow/',
                                            FileName=FileList,
                                            SpanAve=True,
                                            Equ=equ,
                                            OutFile=dirs[i])

                if i == 0:
                    flow = df
                else:
                    flow = pd.concat([flow, df])

        flow = flow.sort_values(by=['x', 'y', 'z'])
        flow = flow.drop_duplicates(keep='last')
        flow.to_hdf(path + 'MeanFlow/' + 'MeanFlow.h5', 'w', format='fixed')
        self._data_field = flow

    @classmethod
    def spanwise_average(cls, path, nfiles):
        dirs = glob(path + 'TP_stat' + '*.plt')
        equ = ['{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)']
        for i in np.range(np.size(dirs)):
            df = p2p.ReadAllINCAResults(nfiles,
                                        path + 'TP_stat/',
                                        path + 'MeanFlow/',
                                        FileName=dirs[i],
                                        SpanAve=True,
                                        Equ=equ,
                                        OutFile='MeanFlow' + str(i))
            return(df)

    def yprofile(self, var_name, var_val):
        df1 = self.PlanarData.loc[self.PlanarData[var_name] == var_val]
        up_boundary = np.max(self.PlanarData['y'])
        down_boundary = np.min(df1['y'])
        if np.max(df1['y']) < up_boundary:
            y_all = np.unique(self.PlanarData['y'])
            y_fill = y_all[y_all > np.max(df1['y'])]
            x_fill = var_val * np.ones(np.size(y_fill))
            xy_fill = np.transpose(np.vstack((x_fill, y_fill)))
            df_fill = pd.DataFrame(data=xy_fill, columns=['x', 'y'])
            df_all = pd.concat((self.PlanarData, df_fill), ignore_index=True)
            df_all = df_all.interpolate(axis='rows')
            df1 = df_all.loc[df_all[var_name] == var_val]
        # df2 = df1.drop_duplicates(subset=['y'], keep='last', inplace=True)
        grouped = df1.groupby(['y'])
        df2 = grouped.mean().reset_index()
        # df1 = PlanarField.uniq1d(df1, 'y')
        df3 = df2.sort_values(by=['y'], ascending=True)
        df4 = df3.loc[df3['y'] >= down_boundary]
        return(df4)

    def add_walldist(self, StepHeight):
        self._data_field['walldist'] = self._data_field['y']
        self._data_field.loc[self._data_field['x'] > 0.0, 'walldist'] \
            += StepHeight

    def add_energy(self):
        tke = 0.5 * (self.R11 + self.R22 + self.R33)
        self._data_field['tke'] = tke
        return(tke)

    def copy_meanval(self, option='Forward'):
        if option == 'Forward':
            self._data_field['u'] = self.u_m
            self._data_field['v'] = self.v_m
            self._data_field['w'] = self.w_m
            self._data_field['rho'] = self.rho_m
            self._data_field['p'] = self.p_m
            self._data_field['T'] = self.T_m
            if 'mu' in self._data_field.columns:
                self._data_field['mu'] = self.mu_m
            if 'vorticity_1' in self._data_field.columns:
                self._data_field['vorticity_1'] = self.vorticity_1_m
            if 'vorticity_2' in self._data_field.columns:
                self._data_field['vorticity_2'] = self.vorticity_2_m
            if 'vorticity_3' in self._data_field.columns:
                self._data_field['vorticity_3'] = self.vorticity_3_m
        else:
            self._data_field['<u>'] = self.u
            self._data_field['<v>'] = self.v
            self._data_field['<w>'] = self.w
            self._data_field['<rho>'] = self.rho
            self._data_field['<p>'] = self.p
            self._data_field['<T>'] = self.T

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:24:50 2017
    This code for reading data from specific file so that post-processing data, including:
    1. FileName (infile, VarName, (row1), (row2), (unique) ): sort data and 
    delete duplicates, SPACE is NOT allowed in VarName
    2. MergeFile (NameStr, FinalFile): NameStr-input file name to be merged
    3. GetMu (T): calculate Mu if not get from file
    4. unique_rows (infile, outfile):
@author: Web-PC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from contextlib import contextmanager
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
import time
import sys
import re
import os
import plt2pandas as p2p

class DataPost(object):
    def __init__(self):
        pass
        self._DataTab = pd.DataFrame()
        self.MeanFlow = pd.DataFrame()
        self.TriFlow = pd.DataFrame()
        self.ProbeSignal = pd.DataFrame()

    def add_variable(frame, var_name, var_val):
        frame[var_name] = var_val


    def LoadData(self, Infile, skiprows = None, \
                 Sep = None, Uniq = True):
        # SkipHeader: skip i-th row, zero-based; 0:skip 1st line; 1:skip 2nd line
        start_time = time.clock()
        assert isinstance(Infile, str), \
        "InputFile:{} is not a string".format(Infile.__class__.__name__)
        #Infile = open(InputFile)
        if Sep is not None:
            self._DataTab = pd.read_csv(Infile, sep=Sep, \
                                        skiprows=skiprows,\
                                        skipinitialspace=True)
        else:
            self._DataTab = pd.read_csv(Infile, sep="\t", \
                                        skiprows=skiprows,\
                                        skipinitialspace=True)
        self._DataTab = self._DataTab.dropna(axis=1, how='all')
        VarInd = ('x' in self._DataTab.columns) & \
                ('y' in self._DataTab.columns) & ('z' in self._DataTab.columns)
        if VarInd:
            self._DataTab = self._DataTab.sort_values(by=['x', 'y', 'z'])
        if 'Solution Time' in self._DataTab.columns:
            self._DataTab = \
            self._DataTab.rename(columns = {'Solution Time':'time'})
            self._DataTab = \
            self._DataTab.sort_values(by=['time'])
        if Uniq == True:
            self._DataTab = self._DataTab.drop_duplicates(keep='last')
        #Infile.close()
        print("The cost time of reading: ", \
            Infile, time.clock()-start_time)

    def AddVariable(self, VarName, VarVal):
        self._DataTab[VarName] = VarVal

    def AddSolutionTime(self, SolutionTime):
        mt = np.shape(SolutionTime)[0]
        m  = np.shape(self._DataTab)[0]
        #assert isinstance(m/mt, int), "Infile:{} is not a string".format(Infile.__class__.__name__)
        time = np.tile(SolutionTime, int(m/mt))
        # tile: copy as a block; repeat: copy every single row separately
        self._DataTab['time'] = time

    def AddWallDist(self, StepHeight):
        self._DataTab['walldist'] = self._DataTab['y']
        self._DataTab.loc[self._DataTab['x'] > 0.0, 'walldist'] += StepHeight

    def AddMu(self, Re_delta):
        mu_unitless = 1.0/Re_delta*np.power(self.T, 0.75)
        self._DataTab['mu'] = mu_unitless

    def AddUGrad(self, WallSpace):
        self._DataTab['UGrad'] = self._DataTab['u']/(0.5*WallSpace)

    def AddMach(self, Ma_inf):
        c = self._DataTab['u']**2+self._DataTab['v']**2+self._DataTab['w']**2
        self._DataTab['Mach'] = Ma_inf * np.sqrt(c/self._DataTab['T'])
    

    @classmethod
    def SutherlandLaw (cls, T, T_inf = None, UnitMode = True):
        if UnitMode is True:
            T_inf = 1.0
        else:
            if T_inf is None:
                print("Must specify temperature of the free stream")
        T = T*T_inf
        T0 = 273.15
        C = 110.4
        mu0 = 1.716e-6
        a = (T0+C)/(T+C)
        b = np.power(T/T0, 1.5)
        mu_unit = mu0*a*b
        return (mu_unit)

    def AddMu_unit(self, T_inf):
        mu_unit = self.SutherlandLaw (self.T, T_inf, UnitMode = False)
        self._DataTab['mu'] = mu_unit
        return (mu_unit)

    def UserData(self, VarName, Infile, SkipHeader, Sep=None, Uniq=True):
        start_time = time.clock()
        #Infile = open(InputFile, 'r')
        self._DataTab = pd.read_csv(Infile, sep = ' ', index_col = False, \
                                    header=None, names=VarName, \
                                    skiprows=SkipHeader, skipinitialspace=True)
        if Sep is not None:
            self._DataTab = pd.read_csv(Infile, sep = Sep, index_col = False,\
                                header=None, names=VarName, \
                                skiprows=SkipHeader, skipinitialspace=True)

        self._DataTab = self._DataTab.dropna(axis=1, how='all')
        if ('x' in VarName) & ('y' in VarName):
            self._DataTab = self._DataTab.sort_values(by=['x', 'y'])
        if ('x' in VarName) & ('y' in VarName) & ('z' in VarName):
            self._DataTab = self._DataTab.sort_values(by=['x', 'y', 'z'])
        if Uniq == True:
            self._DataTab = self._DataTab.drop_duplicates(keep='last')
        print("The cost time of reading: ", Infile, time.clock()-start_time)
        #Infile.close()

    def UserDataBin(self, Infile, Uniq=True, VarName=None):
        self._DataTab = pd.read_hdf(Infile)
        self._DataTab = self._DataTab.dropna(axis=1, how='all')
        if Uniq == True:
            self._DataTab = self._DataTab.drop_duplicates(keep='last')
        if VarName is not None:
            self._DataTab.columns = VarName
        else:
            VarName = list(self._DataTab.columns.values)
        if ('z' in VarName):
            self._DataTab = self._DataTab.sort_values(by=['x', 'y', 'z'])
        else:
            self._DataTab = self._DataTab.sort_values(by=['x', 'y'])


#   Obtain the filename of the probe at a specific location
    def GetProbeName (self, xx, yy, zz, path):
        Infile = open(path + 'inca_probes.inp')
        Prob = np.loadtxt(Infile, skiprows = 6, \
                                usecols = (1,2,3,4))
        Infile.close()
        xarr = np.around(Prob[:,1], 3)
        yarr = np.around(Prob[:,2], 3)
        zarr = np.around(Prob[:,3], 3)
        ProbIndArr = np.where((xarr[:]==np.around(xx,3))\
                              & (yarr[:]==np.around(yy,3)) \
                              & (zarr[:]==np.around(zz,3)))
        if len(ProbIndArr[0]) == 0:
            print ("The probe you input is not found in the probe files!!!")
        ProbInd = ProbIndArr[0][-1] + 1
        FileName = 'probe_'+format(ProbInd, '05d')+'.dat'
        return (FileName)

#%% Read probe data from the INCA results
    def LoadProbeData (self, xx, yy, zz, path, Uniq=True):
        varname = ['itstep', 'time', 'u', 'v', 'w', 'rho', 'E', 'walldist', 'p']
        FileName = self.GetProbeName(xx, yy, zz, path)
        print('Probe file name:', FileName)
        self.UserData(varname, path + FileName, 2)
        self._DataTab = self._DataTab.sort_values(by=['time'])
        if Uniq == True:
            self._DataTab = self._DataTab.drop_duplicates(keep='last')

#   Make DataTab unchangable for users
    @property
    def DataTab(self):
        return self._DataTab
#    By this method to change the DataMat
#    @DataMat.setter
#    def DataMat(self, DataMat):
#        self._DataMat = DataMat
    @property
    def time(self):
        return self._DataTab['time'].values

    @property
    def x(self):
        return self._DataTab['x'].values

    @property
    def y(self):
        return self._DataTab['y'].values

    @property
    def z(self):
        return self._DataTab['z'].values

    @property
    def u(self):
        return self._DataTab['u'].values

    @property
    def u_m(self):
        return self._DataTab['<u>'].values

    @property
    def v(self):
        return self._DataTab['v'].values

    @property
    def v_m(self):
        return self._DataTab['<v>'].values

    @property
    def w(self):
        return self._DataTab['w'].values

    @property
    def w_m(self):
        return self._DataTab['<w>'].values

    @property
    def rho(self):
        return self._DataTab['rho'].values

    @property
    def rho_m(self):
        return self._DataTab['<rho>'].values

    @property
    def p(self):
        return self._DataTab['p'].values

    @property
    def p_m(self):
        return self._DataTab['<p>'].values

    @property
    def Mach(self):
        return self._DataTab['Mach'].values

    @property
    def T(self):
        return self._DataTab['T'].values

    @property
    def mu(self):
        return self._DataTab['mu'].values

    @property
    def mu_m(self):
        return self._DataTab['<mu>'].values

    @property
    def walldist(self):
        return self._DataTab['walldist'].values

    @property
    def uu(self):
        return self._DataTab['<u`u`>'].values
    @property
    def vv(self):
        return self._DataTab['<v`v`>'].values

    @property
    def ww(self):
        return self._DataTab['<w`w`>'].values

    @property
    def uv(self):
        return self._DataTab['<u`v`>'].values

    @property
    def uw(self):
        return self._DataTab['<u`w`>'].values

    @property
    def vw(self):
        return self._DataTab['<v`w`>'].values

    @property
    def UGrad(self):
        return self._DataTab['UGrad'].values
    
    @property
    def rhoGrad(self):
        return self._DataTab['|grad(rho)|'].values

    @property
    def vorticity_1(self):
        return self._DataTab['vorticity_1'].values
    @property
    def vorticity_2(self):
        return self._DataTab['vorticity_2'].values
    @property
    def vorticity_3(self):
        return self._DataTab['vorticity_3'].values
    @property
    def Qcrit(self):
        return self._DataTab['Q-criterion'].values
    @property
    def L2crit(self):
        return self._DataTab['L2-criterion'].values

    def LoadMeanFlow(self, path, nfiles=44):
        exists = os.path.isfile(path + 'MeanFlow/MeanFlow.h5')
        if exists:
            self._DataTab = pd.read_hdf(path + 'MeanFlow/MeanFlow.h5')
        else:
            df = p2p.ReadAllINCAResults(nfiles, path + 'TP_stat/',
                                        path + 'MeanFlow/',
                                        SpanAve=True,
                                        OutFile='MeanFlow')
            self._DataTab = df
        
#   Obtain variables profile with an equal value (x, y, or z)
    def IsoProfile2D (self, qx, Iso_qx, qy):
        assert isinstance(qx, str), \
            "qx:{} is not a string".format(qx.__class__.__name__)
        assert isinstance(qy, str), \
            "qy:{} is not a string".format(qy.__class__.__name__)
        qval = self._DataTab.loc[self._DataTab[qx] == Iso_qx, qy]
        xval = self._DataTab.loc[self._DataTab[qx] == Iso_qx, qx]
        return (xval, qval)

    #   Obtain variables profile with an equal array (x,y)
    def IsoProfile3D (self, qx, Iso_qx, qz, Iso_qz, qy):
        assert isinstance(qx, str), \
            "qx:{} is not a string".format(qx.__class__.__name__)
        assert isinstance(qz, str), \
            "qz:{} is not a string".format(qz.__class__.__name__)
        assert isinstance(qy, str), \
            "qy:{} is not a string".format(qy.__class__.__name__)
        DataTab1  = self._DataTab.loc[self._DataTab[qx] == Iso_qx]
        xval  = DataTab1.loc[DataTab1[qz] == Iso_qz, qx]
        zval  = DataTab1.loc[DataTab1[qz] == Iso_qz, qz]
        qval  = DataTab1.loc[DataTab1[qz] == Iso_qz, qy]
        return (xval, zval, qval)
#        index = np.where (qx[:] == EquVal)
#        if (np.size(index) == 0):    # xval does not exist
#            yy = np.unique (qx)
#            index = np.where (yy[:] > EquVal)    # return index whose value > xval
#            y1 = yy[index[0][0]-1]    # obtain the last value  < xval
#            y2 = yy[index[0][0]]    # obtain the first value > xval
#            index1 = np.where (qx[:] == y1)    # get their index
#            index2 = np.where (qx[:] == y2)
#            qval = (qx[index1] + qx[index2])/2.0
#            xval = (qy[index1] + qy[index2])/2.0
#        else:
#            qval = qy[index]
#            xval = qx[index]
#        if mode is None:
#            return (xval, qval)
#        else:
#            return (qval)

#   Lagrange Interpolation
    @classmethod
    def InterpLagra (cls, xval, x1, x2, q1, q2):
        a1 = (xval-x2)/(x1-x2)
        a2 = (xval-x1)/(x2-x1)
        qval = q1*a1 + q2*a2
        return (qval)

#            a1 = (xval - x1)/(x2 - x1)
#            a2 = (x2 - xval)/(x2 - x1)
#            qval = qy[index1]*a1 + qy[index2]*a2
#   Obtan BL Profile at a Certain X or Z Valude
#   q: the desired quantity; mode 2: return qy and walldist, or only return qy
    def BLProfile (self, qx, Iso_qx, qy):
        qval  = self.IsoProfile2D(qx, Iso_qx, qy)[1]
        WDval = self.IsoProfile2D(qx, Iso_qx, 'walldist')[1]
        if (np.size(qval) == 0 or np.size(WDval) == 0):
            x1 = self._DataTab.loc[self._DataTab[qx] < Iso_qx, \
                                   qx].tail(1).values[0]
            x2 = self._DataTab.loc[self._DataTab[qx] > Iso_qx, \
                                   qx].head(1).values[0]
            DataTab1 = self._DataTab.loc[self._DataTab[qx] == x1]
            DataTab2 = self._DataTab.loc[self._DataTab[qx] == x2]
            wd1   = DataTab1['walldist'].values
            qval1 = DataTab1[qy].values
            wd2   = DataTab2['walldist'].values
            qval2 = DataTab2[qy].values
            del WDval, qval
            if (np.max(wd1) < np.max(wd2) or np.size(wd1) < np.size(wd2)):
                qval1 = np.interp(wd2, wd1, qval1)
                WDval = wd2
            elif(np.max(wd1) > np.max(wd2) or np.size(wd1) > np.size(wd2)):
                qval2 = np.interp(wd1, wd2, qval2)
                WDval = wd1
            else:
                WDval = wd1
            qval = self.InterpLagra(Iso_qx, x1, x2, qval1, qval2)
        return (WDval, qval)
#        index = np.where (qx[:] == xval)
#        if (np.size(index) == 0):    # xval does not exist, need to interpolate
#            xx = np.unique (qx)
#            index = np.where (xx[:] > xval)    # return index whose value > xval
#            x1 = xx[index[0]-1]    # obtain the last value  < xval
#            x2 = xx[index[0]]    # obtain the first value > xval
#            index1 = np.where (qx[:] == x1)    # get their index
#            index2 = np.where (qx[:] == x2)
#            a1 = (xval - x1)/(x2 - x1)
#            a2 = (x2 - xval)/(x2 - x1)
#            qval = qy[index1]*a1 + qy[index2]*a2
#            #yval = (y[index1] + y[index2])/2.0
#            WDval = (self.walldist[index1] + self.walldist[index2])/2.0
#        else:
#            qval = qy[index]
#            #yval = y[index]
#            WDval = self.walldist[index]
#
#            return qval, WDval
#        if (WDval[-1] < 6.0):    # y points is not enough to get profile
#            l = locals ()
#            j = 0
#            xvalF = xval
#            while True:
#                xvalF = xvalF - 0.25
#                l['indexF%s' % j] = np.where (qx[:] == xvalF)
#                l['WDvalF%s' % j] = (self.walldist[l['indexF%s' % j]])
#                if (l['WDvalF%s' % j][-1] >= 8.0):
#                    break
#                j = j + 1
#            xvalB = xval
#            i = 0
#            while True:
#                xvalB = xvalB + 0.25
#                l['indexB%s' % i] = np.where (qx[:] == xvalB)
#                l['WDvalB%s' % i] = (self.walldist[l['indexB%s' % i]])
#                if (l['WDvalB%s' % i][-1] >= 8.0):
#                    break
#                i = i + 1
#            indexF = l['indexF%s' % j]
#            indexB = l['indexB%s' % i]
#            qvalF = qy[indexF]
#            qvalB = qy[indexB]
#            WDF = self.walldist[indexF]
#            WDB = self.walldist[indexB]
#            # interpolate walldist to get the same length of array
#            WDval = np.arange (0, 8, 0.02)
#            qval1 = np.interp (WDval, WDF, qvalF)
#            qval2 = np.interp (WDval, WDB, qvalB)
#            #func1 = interp1d (WDF, qvalF, kind = 'linear', fill_value = 'extrapolate')
#            #qval1 = func1 (WDval)
#            a1 = (xval - xvalF)/(xvalB - xvalF)
#            a2 = (xvalB - xval)/(xvalB - xvalF)
#            qval = qval1*a1 + qval2*a2
#        return (WDval, qval)

#   Obtain Average Value of Data with Same Coordinates (time-averaged)
    def TimeAve (self):
        grouped = self._DataTab.groupby([self._DataTab['x'], \
                                        self._DataTab['y'], self._DataTab['z']])
        self._DataTab = grouped.mean().reset_index(drop=True)


#   Extract interesting part of the data
    def ExtraPoint (self, Mode, Val):
        self._DataTab = self._DataTab.loc[self._DataTab[Mode] >= Val]
        self._DataTab = self._DataTab.head(1)

    def ExtraSeries (self, Mode, Min, Max):
        self._DataTab = self._DataTab.loc[self._DataTab[Mode] >= Min]
        self._DataTab = self._DataTab.loc[self._DataTab[Mode] <= Max]

#   Obtain finite differential derivatives of a variable (2nd order)
    @classmethod
    def SecOrdFDD (cls, xarr, var):
        dvar = np.zeros(np.size(xarr))
        for jj in range (1,np.size(xarr)):
            if jj == 1:
                dvar[jj-1]=(var[jj]-var[jj-1])/(xarr[jj]-xarr[jj-1])
            elif jj == np.size(xarr):
                dvar[jj-1]=(var[jj-1]-var[jj-2])/(xarr[jj-1]-xarr[jj-2])
            else:
                dy12 = xarr[jj-1] - xarr[jj-2];
                dy23 = xarr[jj] - xarr[jj-1];
                dvar1 = -dy23/dy12/(dy23+dy12)*var[jj-2];
                dvar2 = (dy23-dy12)/dy23/dy12*var[jj-1];
                dvar3 = dy12/dy23/(dy23+dy12)*var[jj];
                dvar[jj-1] = dvar1 + dvar2 + dvar3;
        return (dvar)

#   Show Progress of code loop
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

# Merge Several Files into One
# Files must have same data structure
# NameStr: name of files waiting for being merged, FinalFile: merged file name
    def MergeFile (self, path, NameStr, FinalFile):
        l = locals ()
        #read data header/title and save
        with open (path + NameStr[0]) as f:
            title1 = f.readline ().split ('\t')
            title2 = f.readline ().split ()
        title = '\n'.join([str(title1), str(title2)])
        f.close()
        #read data
        unique_rows(NameStr[0], FinalFile)
        Infile = open(path + FinalFile)
        Data = np.loadtxt (Infile, skiprows = 2)
        Infile.close()
        n = len(NameStr)
        #merge the rest of files
        for j in range (1, n):
            unique_rows(NameStr[j], FinalFile)
            Infile = open(path + FinalFile)
            l['Data'+str(j)] = np.loadtxt (Infile, skiprows = 2)
            Infile.close()
            Data = np.concatenate ([Data, l['Data'+str(j)] ])
        Outfile = open(path+FinalFilw)
        np.savetxt (Outfile, Data, \
                    fmt='%1.8e', delimiter = "\t", header = str(title))
        Outfile.close()
        print ("Congratulations! Successfully obtain <" \
               +FinalFile+ "> by merging files:")
        print (NameStr)

#   Obtain Spanwise Average Value of Data
    def SpanAve(self, OutputFile = None):
        start_time = time.clock()
        grouped = self._DataTab.groupby(['x', 'y'])
        self._DataTab = grouped.mean().reset_index()
        if OutputFile is not None:
            outfile  = open(OutputFile, 'x')
            self._DataTab.to_csv(outfile, \
                            index=False, sep = '\t')
            outfile.close()
        print("The computational time is ", time.clock()-start_time)


#   Detect peaks in data based on their amplitude and other features.
    @classmethod
    def FindPeaks(cls, x, mph=None, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False, show=False, ax=None):

        """Detect peaks in data based on their amplitude and other features.
    
        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).
    
        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.
        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`
        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            plot(x, mph, mpd, threshold, edge, valley, ax, ind)
        return ind

#   Obtain growth rate of variables in a specific drection (x, y, z or time)
#   both var and Ampli are 0-based, means that their average are zero
    @classmethod
    def GrowthRate (cls, xarr, var, Mode = None):
        if Mode is None:
            # xarr is x-coordinates, var is corresponding value of variable
            # this part is to calculate the growth rate according to
            # exact value of variable with the x value
            AmpInd = cls.FindPeaks(var)
            AmpVal = var[AmpInd]
            xval   = xarr[AmpInd]
            dAmpdx = cls.SecOrdFDD(xval, AmpVal)
            growthrate = dAmpdx/AmpVal
            return (xval, growthrate)
        else:
            # this part is to calculate the growth rate according to
            # amplitude of variable with the x value
            dAmpli = cls.SecOrdFDD(xarr, var)
            growthrate = dAmpli/var
            xval = xarr
            return growthrate

#   Get Frequency Weighted Power Spectral Density
    @classmethod
    def FW_PSD (cls, VarZone, TimeZone, pic = None):
        var = VarZone
        ave = np.mean (var)
        var_fluc = var-ave
        #    fast fourier transform and remove the half
        var_fft = np.fft.rfft (var_fluc)
        var_psd = abs(var_fft)**2
        num = np.size (var_fft)
        #    sample frequency
        fre_samp = num/(TimeZone[-1]-TimeZone[0])
        #f_var = np.linspace (0, f_samp/2, num)
        fre = np.linspace (fre_samp/2/num, fre_samp/2, num)
        fre_weighted = var_psd*fre
        if pic is not None:
            fig, ax = plt.subplots ()
            ax.semilogx (fre, fre_weighted)
            ax.ticklabel_format (axis = 'y', style = 'sci', scilimits = (-2, 2))
            ax.set_xlabel (r'$f\delta_0/U_\infty$', fontdict = font2)
            ax.set_ylabel ('Weighted PSD, unitless', fontdict = font2)
            ax.grid (b=True, which = 'both', linestyle = '--')
            fig.savefig (pic, dpi = 600)
            #plt.show ()
        return (fre, fre_weighted)

#   Detect local peaks in a vector
    @classmethod
    def peakdet (cls, v, delta, x = None):
        """
        Returns two arrays [MAXTAB, MINTAB] 
        MAXTAB and MINTAB consists of two columns. (1:indices, 2:values).
        A point is considered a maximum peak if it has the maximal
        value, and was preceded (to the left) by a value lower by DELTA.
        """
        maxtab = []
        mintab = []

        if x is None:
            x = arange(len(v))

        v = asarray(v)

        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')

        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')

        if delta <= 0:
            sys.exit('Input argument delta must be positive')

        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN

        lookformax = True

        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]

            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        return array(maxtab), array(mintab)

#   fit data using sinusoidal functions
    def fit_sin(cls, tt, yy, guess_omeg):
        '''Fit sin to the input time sequence, and return fitting parameters
        "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        # w, excluding the zero frequency "peak", which is related to offset
        #guess_freq = 1.147607 #abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(yy) * 2.**0.5 # A
        guess_offset = np.mean(yy)   # c
        guess = np.array([guess_amp, guess_freq, 0., guess_offset])  # p=0
        #guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])  # p=0
        # define the fitting function
        def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        #f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p) + c
        return {"amp": A, "omega": w, "phase": p, "offset": c, \
                "freq": w/2/np.pi, "period": 2*np.pi/w, "fitfunc": fitfunc, \
                "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

#   fit data using a specific functions
    @classmethod
    def fit_func(cls, function, tt, yy, guess=None):
        if guess is not None:
            popt, pcov = scipy.optimize.curve_fit(function, tt, yy, p0 = guess)
        else:
            popt, pcov = scipy.optimize.curve_fit(function, tt, yy, absolute_sigma=True)
        return (popt, pcov)
        #return{"coeff": popt, "rawres": (guess, popt, pcov)}

#   fit data using sinusoidal functions
    @classmethod
    def fit_sin2(cls, tt, yy, guess_freq):
        '''Fit sin to the input time sequence, and return fitting parameters
        "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        #        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        #        Fyy = abs(np.fft.fft(yy))
        # w, excluding the zero frequency "peak", which is related to offset
        #guess_freq = 1.147607 #abs(ff[np.argmax(Fyy[1:])+1])
        guess_amp = np.std(yy) * 2.**0.5 # A
        #guess_offset = np.mean(yy)   # c
        guess_phase = 0.0
        guess = np.array([guess_amp, guess_freq, guess_phase])  # p=0
        #guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])  # p=0
        # define the fitting function
        def sinfunc(t, A, w, p):  return A * np.sin(w*t + p)
        popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p = popt
        #f = w/(2.*np.pi)
        fitfunc = lambda t: A * np.sin(w*t + p)
        return {"amp": A, "omega": w, "phase": p, \
                "freq": w/2/np.pi, "period": 2*np.pi/w, "fitfunc": fitfunc, \
                "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
    @classmethod
    def ReadProbeXYZ(cls, Infile):
        FirstLine = open(Infile).readline()
        xyz       = re.findall(r"[+-]?\d+\.?\d*", FirstLine)
        print(xyz)
        return xyz

## backup code for data fitting
#guess_freq = 1
#guess_amplitude = 3*np.std(data)/(2**0.5)
#guess_phase = 0
#guess_offset = np.mean(data)
#
#p0=[guess_freq, guess_amplitude,
#    guess_phase, guess_offset]
#
## create the function we want to fit
#def my_sin(x, freq, amplitude, phase, offset):
#    return np.sin(x * freq + phase) * amplitude + offset
#
## now do the fit
#fit = curve_fit(my_sin, t, data, p0=p0)
#
## we'll use this to plot our first estimate. This might already be good enough for you
#data_first_guess = my_sin(t, *p0)

# recreate the fitted curve using the optimized parameters
#data_fit = my_sin(t, *fit[0])

#if __name__ == "__main__":
#    a = DataPost()
#    path = "../../TestData/"
#    #    ind = a.LoadProbeData (80.0, 0.0, 0.0, path)
#    a.LoadData(path + 'TimeSeries2X0Y0Z0.txt')
#    b = DataPost()
#    b.LoadData(path + 'Time1600Z0Slice.txt', skiprows = [1])
#    b.AddWallDist(3.0)
#    bl = b.BLProfile('x', 0.3, 'u')
#    plt.plot(bl[1], bl[0])
#    plt.show()
#    #    qval = b.IsoProfile3D('x', 0.0, 'z', 0.0, 'Mach')
#    qval = b.IsoProfile2D('x', 0.0, 'u')
#    c = DataPost()
#    c.LoadProbeData(0.0, 0.0, -2.7, path, Uniq = False)
#
#    path1="/media/weibo/Data1/BFS_M1.7L_0419/probes/old/"
#    path2="/media/weibo/Data1/BFS_M1.7L_0419/probes/"
#    path3="/media/weibo/Data1/BFS_M1.7L_0419/probes/new/"
#    with open(path1+"probe_00238.dat") as f1:
#        Data1 = np.loadtxt(f1, skiprows=0)
#    with open(path2+"probe_00238.dat") as f2:
#        Data2 = np.loadtxt(f2, skiprows=0)
#    Data = np.concatenate([Data1, Data2])
#    np.savetxt(path3+"probe_00238.dat", Data, fmt='%1.8e', delimiter = "  ")
#    print("Congratulations!")




#    #a.GetMu_nondim (3856)
#    a.Getwalldist (3.0)
#    a.unique_rows()
#    z, var = a.EquValProfile(0.0, a.y, a.p)
#    VarName = ['NO', 'time', 'u', 'v', 'w', 'rho', 'E', 'walldist', 'p']
#    a.UserData(VarName, 'probe_00068.dat', 1)

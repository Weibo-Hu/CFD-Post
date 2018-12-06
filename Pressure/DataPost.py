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
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import spline
import scipy.optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
import sys

class DataPost(object):
    def __init__(self):
        pass
        self._DataMat = [None]*13
        self._DataTab = pd.DataFrame()
#        self._DataMat = self._DataMat.astype(np.float64)

    def LoadData(self, Infile, HeadLine, TimeMode):
        start_time = time.clock()
        assert isinstance(Infile, str), "Infile:{} is not a string".format(Infile.__class__.__name__)
        self._DataTab = pd.read_csv(Infile, sep="\t", skiprows=HeadLine, \
                                    skipinitialspace=True)
        self._DataTab = self._DataTab.dropna(axis=1, how='all')
        self._DataTab = self._DataTab.sort_values(by=['x', 'y', 'z'])
        if TimeMode == 'time':
            self._DataMat = self._DataTab.values
            #self._DataMat = self._DataMat[\
            #    np.lexsort((self._DataMat[:,3], self._DataMat[:,2], self._DataMat[:,1], self._DataMat[:,0]))]
        else:
            m, n = self._DataTab.shape
#            try:
#                n == 9
#            except IOError:
#                print ("The data format does not match! It should be")
            time_normalized = np.full ((m,1), TimeMode)
            self._DataMat = np.concatenate ((time_normalized, self._DataTab.values), axis =1)
            #self._DataMat = np.unique (self._DataMat, axis = 0)
            del time_normalized
            #self._DataMat = self._DataMat[\
            #   np.lexsort((self._DataMat[:,3], self._DataMat[:,2], self._DataMat[:,1]))]
        print("The cost time of reading: ", Infile, time.clock()-start_time)

    def LoadData1(self, Infile, HeadLine, TimeMode):
        start_time = time.clock()
        assert isinstance(Infile, str), "Infile:{} is not a string".format(Infile.__class__.__name__)
        if TimeMode == 'time':
            self._DataMat = np.loadtxt(Infile, skiprows = HeadLine)
            #self._DataMat = np.unique (self._DataMat, axis = 0)
#            assert isinstance(np.shape(self._DataMat)[1] != 11), \
#            "The data format does not match! It should be \n" + \
#            "time, x, y, z, u, v, w, rho, p, Ma, T"
            self._DataMat = self._DataMat[\
                np.lexsort((self._DataMat[:,3], self._DataMat[:,2], self._DataMat[:,1], self._DataMat[:,0]))]
        else:
            Data = np.loadtxt(Infile, skiprows = HeadLine)
            m, n = Data.shape
#            try:
#                n == 9
#            except IOError:
#                print ("The data format does not match! It should be")
            time_normalized = np.full ((m,1), TimeMode)
            self._DataMat = np.concatenate ((time_normalized, Data), axis =1)
            #self._DataMat = np.unique (self._DataMat, axis = 0)
            del Data, time_normalized
            self._DataMat = self._DataMat[\
                np.lexsort((self._DataMat[:,3], self._DataMat[:,2], self._DataMat[:,1]))]
        print("The cost time of reading: ", Infile, time.clock()-start_time)
        #self._DataMat = np.unique (self._DataMat, axis = 0)
#        if uni_row is not None:    # delete repetitive rows
#            unique_rows (infile, "UniqueData.txt")
#            self._DataMat = np.loadtxt(Infile, skiprows = 2)
#        else:  # default: nothing, i.e. keep repetitive rows
#            self._DataMat = np.loadtxt (Infile, skiprows = 2)
    #   Obtain Nondimensional Dynaimc Viscosity
    def UserData(self, VarName, Infile, HeadLine):
        #RawData = np.loadtxt(Infile, skiprows = HeadLine)
        start_time = time.clock()
        self._DataTab = pd.read_csv(Infile, sep = ' ', \
                                    header = None, names=VarName, \
                                    skiprows=HeadLine, skipinitialspace=True)
        self._DataTab = self._DataTab.dropna(axis=1, how='all')
        RawData = self._DataTab.values
        m, n = RawData.shape
        print(m, n)
        self._DataMat = np.zeros((m,13))
        if 'time' in VarName:
            ind = VarName.index('time')
            self._DataMat[:,0] = RawData[:,ind]
        if 'x' in VarName:
            ind = VarName.index('x')
            self._DataMat[:,1] = RawData[:,ind]
        if 'y' in VarName:
            ind = VarName.index('y')
            self._DataMat[:,2] = RawData[:,ind]
        if 'z' in VarName:
            ind = VarName.index('z')
            self._DataMat[:,3] = RawData[:,ind]
        if 'u' in VarName:
            ind = VarName.index('u')
            self._DataMat[:,4] = RawData[:,ind]
        if 'v' in VarName:
            ind = VarName.index('v')
            self._DataMat[:,5] = RawData[:,ind]
        if 'w' in VarName:
            ind = VarName.index('w')
            self._DataMat[:,6] = RawData[:,ind]
        if 'rho' in VarName:
            ind = VarName.index('rho')
            self._DataMat[:,7] = RawData[:,ind]
        if 'p' in VarName:
            ind = VarName.index('p')
            self._DataMat[:,8] = RawData[:,ind]
        if 'Ma' in VarName:
            ind = VarName.index('Ma')
            self._DataMat[:,9] = RawData[:,ind]
        if 'T' in VarName:
            ind = VarName.index('T')
            self._DataMat[:,10] = RawData[:,ind]
        if 'mu' in VarName:
            ind = VarName.index('mu')
            self._DataMat[:,11] = RawData[:,ind]
        if 'WallDist' in VarName:
            ind = VarName.index('WallDist')
            self._DataMat[:,12] = RawData[:,ind]
        self._DataMat = np.unique (self._DataMat, axis = 0)
        print("The cost time of reading: ", Infile, time.clock()-start_time)
#   Make DataMat unchangable for users
    @property
    def DataMat(self):
        return self._DataMat

    @property
    def DataTab(self):
        return self._DataTab
#    By this method to change the DataMat
#    @DataMat.setter
#    def DataMat(self, DataMat):
#        self._DataMat = DataMat
        
    @property
    def x(self):
        #return self.DataMat[:,1]
        return self.DataTab['x']
    
    @property
    def y(self):
        #return self.DataMat[:,2]
        return self.DataTab['y']
    
    @property
    def z(self):
        return self.DataTab['z']
    
    @property
    def u(self):
        return self.DataTab['u']
    
    @property
    def v(self):
        return self.DataTab['v']
    
    @property
    def w(self):
        return self.DataTab['w']
    
    @property
    def rho(self):
        return self.DataTab['rho']
    
    @property
    def p(self):
        return self.DataTab['p']
    
    @property
    def Ma(self):
        return self.DataTab['Ma']
    
    @property
    def T(self):
        return self.DataTab['T']
    
    @property
    def mu(self):
        return self.DataTab['mu']
    
    @property
    def WallDist(self):
        return self.DataTab['WallDist']

    @property
    def UGrad(self):
        return self.DataTab['u']/0.0078125
    
#   Obtain nondimensional dynamic viscosity
    def GetMu_nondim(self, Re_delta):
        mu_nondim = 1.0/Re_delta*np.power (self.T, 0.75)
        mu_nondim = np.reshape (mu_nondim, (np.size (mu_nondim), 1))
        self._DataMat = np.hstack ((self._DataMat, mu_nondim))

#   Obtain nondimensional wall distance
    def GetWallDist(self, StepHeight):
        m = np.shape(self._DataMat)[0]
        WallDist = np.zeros(shape = (m,1))
        for j in range (m):
            if self.x[j] <= 0.0:
                WallDist[j] = self.y[j]
            else:
                WallDist[j] = self.y[j] + StepHeight
        mu_nondim = np.zeros(shape = (m,1))
        self._DataMat = np.hstack((self._DataMat, mu_nondim, WallDist))
        
#   Obatin solution time
    def GetSolutionTime(self, SolutionTime):
        mt = np.shape(SolutionTime)[0]
        m  = np.shape(self._DataMat)[0]
        #assert isinstance(m/mt, int), "Infile:{} is not a string".format(Infile.__class__.__name__)
        time = np.tile(SolutionTime, int(m/mt))
        #time = np.repeat(SolutionTime, int(m/mt))
        #print (np.shape(time))
        self._DataMat[:,0] = time
        
#   Show Progress of code loop
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
#   Remove Duplicated Rows
    def unique_rows(self):
        self._DataMat = np.unique (self._DataMat, axis = 0)
        
#   Obtain variables profile with an equal array (x,y)
    def EquValProfile3D (self, var, x=None, y=None):
        if (x is not None) & (y is not None):
            ind = np.where(self.x[:] == x)
            yval = self.y[ind]
            zval = self.z[ind]
            q    = var[ind]
            ind1 = np.where(yval[:] == y)
            q0   = q[ind1]
            z0   = zval[ind1]
        return (z0, q0)
#   Obtain variables profile with an equal value (x, y, or z)
    def EquValProfile (self, EquVal, qx, qy, mode = None):
        index = np.where (qx[:] == EquVal)
        if (np.size(index) == 0):    # xval does not exist
            yy = np.unique (qx)
            index = np.where (yy[:] > EquVal)    # return index whose value > xval
            y1 = yy[index[0][0]-1]    # obtain the last value  < xval
            y2 = yy[index[0][0]]    # obtain the first value > xval
            index1 = np.where (qx[:] == y1)    # get their index
            index2 = np.where (qx[:] == y2)
            qval = (qx[index1] + qx[index2])/2.0
            xval = (qy[index1] + qy[index2])/2.0
        else:
            qval = qy[index]
            xval = qx[index]
        if mode is None:
            return (xval, qval)
        else:
            return (qval)
#   Obtan BL Profile at a Certain X or Z Valude
#   q: the desired quantity; mode 2: return qy and walldist, or only return qy
    def BLProfile (self, xval, qx, qy, mode = None):
        index = np.where (qx[:] == xval)
        if (np.size(index) == 0):    # xval does not exist, need to interpolate
            xx = np.unique (qx)
            index = np.where (xx[:] > xval)    # return index whose value > xval
            x1 = xx[index[0]-1]    # obtain the last value  < xval
            x2 = xx[index[0]]    # obtain the first value > xval        
            index1 = np.where (qx[:] == x1)    # get their index
            index2 = np.where (qx[:] == x2)
            a1 = (xval - x1)/(x2 - x1)
            a2 = (x2 - xval)/(x2 - x1)
            qval = qy[index1]*a1 + qy[index2]*a2
            #yval = (y[index1] + y[index2])/2.0
            WDval = (self.WallDist[index1] + self.WallDist[index2])/2.0
        else:
            qval = qy[index]
            #yval = y[index]
            WDval = self.WallDist[index]
            
            return qval, WDval
            
        if (WDval[-1] < 6.0):    # y points is not enough to get profile
            l = locals ()
            j = 0
            xvalF = xval
            while True:
                xvalF = xvalF - 0.25
                l['indexF%s' % j] = np.where (qx[:] == xvalF)
                l['WDvalF%s' % j] = (self.WallDist[l['indexF%s' % j]])
                if (l['WDvalF%s' % j][-1] >= 8.0):
                    break
                j = j + 1
            xvalB = xval
            i = 0
            while True:
                xvalB = xvalB + 0.25
                l['indexB%s' % i] = np.where (qx[:] == xvalB)
                l['WDvalB%s' % i] = (self.WallDist[l['indexB%s' % i]])
                if (l['WDvalB%s' % i][-1] >= 8.0):
                    break
                i = i + 1
            indexF = l['indexF%s' % j]
            indexB = l['indexB%s' % i]
            qvalF = qy[indexF]
            qvalB = qy[indexB]
            WDF = self.WallDist[indexF]
            WDB = self.WallDist[indexB]
            # interpolate WallDist to get the same length of array
            WDval = np.arange (0, 8, 0.02)
            qval1 = np.interp (WDval, WDF, qvalF)
            qval2 = np.interp (WDval, WDB, qvalB)
            #func1 = interp1d (WDF, qvalF, kind = 'linear', fill_value = 'extrapolate')
            #qval1 = func1 (WDval)
            a1 = (xval - xvalF)/(xvalB - xvalF)
            a2 = (xvalB - xval)/(xvalB - xvalF)
            qval = qval1*a1 + qval2*a2
            
        if mode is None:
            return qval, WDval
        else:
            return (qval)
        
#   Obtain Real Dynamic Viscoisty
    def GetMu (T, T_inf):
        T = T * T_inf
        g = globals ()
        T0   = 273.15
        C    = 110.4
        mu0 = 1.716e-6
        a = (T0+C)/(T+C)
        b = np.power (T/T0, 1.5)
        g['mu'] = mu0*a*b
        return (mu)
    

#   Obtain the filename of the probe at a specific location
    def GetProbeName (self, xx, yy, zz, path):
        Prob = np.loadtxt (path+'inca_probes.inp', skiprows = 6, \
                                usecols = (1,2,3,4))
        xarr = Prob[:,1]
        yarr = Prob[:,2]
        zarr = Prob[:,3]
        ProbIndArr = np.where((xarr[:]==xx) & (yarr[:]==yy) & (zarr[:]==zz))
        if len(ProbIndArr[0]) == 0:
            print ("The probe you input is not found in the probe files!!!")
        ProbInd = ProbIndArr[0][-1] + 1
        FileName = 'probe_'+format(ProbInd, '05d')+'.dat'
        return (FileName)
#   Read probe data from the INCA results
    def LoadProbeData (self, xx, yy, zz, path):
        R = 287.05
        gamma = 1.4
        Ma_inf = 3.4
        varname = ['itstep', 'time', 'u', 'v', 'w', 'rho', 'E', 'WallDist', 'p']
        FileName = self.GetProbeName (xx, yy, zz, path)
        #self.UserData (varname, path + FileName, 1)
        #time_uni, ind = np.unique(self.time, return_index=True)
        #self._DataMat = self._DataMat[ind,:]
        m = np.shape(self._DataMat)[0]
        self._DataMat[:,1] = np.tile(xx, m)
        self._DataMat[:,2] = np.tile(yy, m) #np.full ((m,1), yy)[:,0]
        self._DataMat[:,3] = np.tile(zz, m)
        self._DataMat[:,10]= Ma_inf*Ma_inf*gamma*self._DataMat[:, 8]/self._DataMat[:,7]
        #np.full ((m,1), zz)[:,0]
        #self._DataMat = self._DataMat[np.lexsort(self._DataMat[:,0])]

# Merge Several Files into One
# Files must have same data structure
# NameStr: name of files waiting for being merged, FinalFile: merged file name
    def MergeFile (self,NameStr, FinalFile):
        l = locals ()
        #read data header/title and save
        with open (path1+NameStr[0]) as f:
            title1 = f.readline ().split ('\t')
            title2 = f.readline ().split ()
        title = '\n'.join([str(title1), str(title2)])
        #read data
        unique_rows (NameStr[0], FinalFile)
        Data = np.loadtxt (path2+FinalFile, skiprows = 2)
        n = len(NameStr)
        #merge the rest of files
        for j in range (1, n):
            unique_rows (NameStr[j], FinalFile)
            l['Data'+str(j)] = np.loadtxt (path2+FinalFile, skiprows = 2)
            Data = np.concatenate ([Data, l['Data'+str(j)] ])
        np.savetxt (path2+FinalFile, Data, \
                    fmt='%1.6e', delimiter = "\t", header = str(title))
        print ("Congratulations! Successfully obtain <" +FinalFile+ "> by merging files:")
        print (NameStr)

#   Obtain Average Value of Data with Same Coordinates
    def AveAtSameXYZ (self, Mode):
        if Mode == 'All':
            Variables = ['time', 'x', 'y', 'z', 'u', 'v', 'w', \
                     'rho', 'p', 'Mach', 'T', 'mu', 'WallDist']
        else:
            Variables = ['time', 'x', 'y', 'z', 'u', 'v', 'w', \
                     'rho', 'p', 'Mach', 'T']
        dataframe = pd.DataFrame(self._DataMat, columns=Variables)
        grouped = dataframe.groupby([ dataframe['time'], dataframe['x'], \
                                     dataframe['y'], dataframe['z'] ])
        AveGroup = grouped.mean().reset_index()
        self._DataMat = AveGroup.values
        
#   Extract interesting part of the data
    def ExtraData (self, Mode, Min, Max):
        if (Mode == 'time'):
            ind = np.where ((self.time >= Min) & (self.time <= Max))
            self._DataMat = self._DataMat[ind,:][0]
#   Obtain Spanwise Average Value of Data
    def SpanAve (self,infile, outfile):
        var = open (path2+infile).readline ().split()
        DataMat = pd.read_table (path2+infile, sep = '\t', index_col = False)
        grouped = DataMat.groupby([ DataMat['# x'], DataMat['y'] ])
        AveGroup = grouped.mean().reset_index()
        #FinalData = AveGroup.reset_index()
        np.savetxt (path2+outfile, AveGroup.values, \
                    fmt='%1.6e', delimiter = '\t', header = str(var))
        
#   Obtain finite differential derivatives of a variable (2nd order accuracy)
    def SecOrdFDD(self, yarr, var):
        dvar = np.zeros(np.size(yarr))
        for jj in range (1,np.size(yarr)):
            if jj == 1:
                dvar[jj-1]=(var[jj]-var[jj-1])/(yarr[jj]-yarr[jj-1])
            elif jj == np.size(yarr):
                dvar[jj-1]=(var[jj-1]-var[jj-2])/(yarr[jj-1]-yarr[jj-2])
            else:
                dy12 = yarr[jj-1] - yarr[jj-2];
                dy23 = yarr[jj] - yarr[jj-1];
                dvar1 = -dy23/dy12/(dy23+dy12)*var[jj-2];
                dvar2 = (dy23-dy12)/dy23/dy12*var[jj-1];
                dvar3 = dy12/dy23/(dy23+dy12)*var[jj];
                dvar[jj-1] = dvar1 + dvar2 + dvar3;
        return (dvar)
    


#   Detect peaks in data based on their amplitude and other features.
    def FindPeaks(self, x, mph=None, mpd=1, threshold=0, edge='rising',
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
            _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    
        return ind

#   Obtain growth rate of variables in a specific drection (x, y, z or time)
    def GrowthRate (self, xarr, var, Mode = None):
        if Mode is None:
        # xarr is x-coordinates, var is corresponding value of variable
        # this part is to calculate the growth rate according to 
        # exact value of variable with the x value
            AmpInd = self.FindPeaks(var)
            AmpVal = var[AmpInd]
            xval   = xarr[AmpInd]
            dAmpdx = self.SecOrdFDD(xval, AmpVal)
            growthrate = dAmpdx/AmpVal
            return (xval, growthrate)
        else:
        # this part is to calculate the growth rate according to 
        # amplitude of variable with the x value
            dAmpli = self.SecOrdFDD(xarr, var)
            growthrate = dAmpli/var
            xval = xarr
            return growthrate

#   Get Frequency Weighted Power Spectral Density
    def FW_PSD (VarZone, TimeZone, pic = None):
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
    def peakdet (self, v, delta, x = None):
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
    def fit_sin(tt, yy, guess_omeg):
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

#   fit data using sinusoidal functions
    def fit_sin2(tt, yy, guess_freq):
        '''Fit sin to the input time sequence, and return fitting parameters
        "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
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
        
if __name__ == "__main__":
    a = DataPost()
    path = "../probes/"
#    ind = a.LoadProbeData (80.0, 0.0, 0.0, path)
    a.LoadData('TimeSeriesX0Y0Z0.txt', 0, 800)
    b = DataPost()
    b.LoadProbeData(0.0, 0.0, 0.0, path)
#    #a.GetMu_nondim (3856)
#    a.GetWallDist (3.0)
#    a.unique_rows()
#    z, var = a.EquValProfile(0.0, a.y, a.p)
#    VarName = ['NO', 'time', 'u', 'v', 'w', 'rho', 'E', 'WallDist', 'p']
#    a.UserData(VarName, 'probe_00068.dat', 1)

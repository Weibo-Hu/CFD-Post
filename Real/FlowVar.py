# -*- coding: utf-8 -*-
"""
Created on Tue May 1 10:24:50 2018
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
import time
import sys

# Obtain intermittency factor from an undisturbed and specific wall pressure
def Intermittency(sigma, WallPre, TimeZone):
    AvePre    = np.mean(WallPre)
    # wall pressure standard deviation of undisturbed BL
    threshold = AvePre+3*sigma
    # Alternative approximate method
    # DynamicP = 0.5*0.371304*469.852**2, ratio = 0.006/(1+0.13*1.7**2)**0.64
    # sigma1 = DynamicP*ratio
    # threshold value for turbulence
    sign      = (WallPre-threshold)/abs(WallPre-threshold)
    sign      = np.maximum(0, sign[:])
    gamma      = np.trapz(sign, TimeZone)/(TimeZone[-1]-TimeZone[0])
    return gamma

# Obtain Frequency-Weighted Power Spectral Density 
def FW_PSD (VarZone, TimeZone, pic = None):
    Ave = np.mean (VarZone)
    Var_fluc = VarZone-Ave
    # fast fourier transform and remove the half
    Var_fft = np.fft.rfft (Var_fluc)
    Var_psd = abs(Var_fft)**2
    num = np.size (Var_fft)
    # sample frequency
    Fre_samp = num/(TimeZone[-1]-TimeZone[0])
    # f_var = np.linspace (0, f_samp/2, num)
    Fre = np.linspace (Fre_samp/2/num, Fre_samp/2, num)
    Fre_weighted = Var_psd*Fre
    return (Fre, Fre_weighted)

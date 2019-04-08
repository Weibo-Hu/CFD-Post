# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 2D data 
@author: Weibo Hu
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

class PlanarField(object):
    def __init__(self):
        pass
        self.PlanarData = pd.DataFrame()

    @property
    def R11(self):
        return self.PlanarData['<u`u`>'].values
    @property
    def R22(self):
        return self.PlanarData['<v`v`>'].values

    @property
    def R33(self):
        return self.PlanarData['<w`w`>'].values

    @property
    def R12(self):
        return self.PlanarData['<u`v`>'].values

    @property
    def R13(self):
        return self.PlanarData['<u`w`>'].values

    @property
    def R23(self):
        return self.PlanarData['<v`w`>'].values

    def load_meanflow(self, path, nfiles=44):
        exists = os.path.isfile(path + 'MeanFlow/MeanFlow.h5')
        if exists:
            self._DataTab = pd.read_hdf(path + 'MeanFlow/MeanFlow.h5')
        else:
            df = p2p.ReadAllINCAResults(nfiles, path + 'TP_stat/',
                                        path + 'MeanFlow/',
                                        SpanAve=True,
                                        OutFile='MeanFlow')
            self._DataTab = df

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:24:50 2019
    This code for postprocessing 1D data 
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

class LineField(object):
    def __init__(self):
        pass
        self._DataTab = pd.DataFrame()
        self.MeanFlow = pd.DataFrame()
        self.TriFlow = pd.DataFrame()
        self.ProbeSignal = pd.DataFrame()


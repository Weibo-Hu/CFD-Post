#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:52:46 2025
    pre-processing Digtial filter data

@author: weibo
"""

# %% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.ticker as ticker

from glob import glob


# %% read input file
path = '/home/weibo/zong/DF/'
bl = np.loadtxt(path + 'M2.1_ReD_55666.dat', skiprows=1, delimiter=',')
bl_profile = np.reshape(bl, (-1, 11))
nms = ['y/delta_99',   'u/u_e', 'v/u_e', 'r/r_e', 'T/T_e', 'uu',
       'vv',  'ww',  'uv', 'rr', 'TT']
bl_profile = pd.DataFrame(bl_profile, columns=nms)
bl_profile['u+'] = 1.0
bl_profile['w/u_e'] = 0.0
bl_profile['uw'] = 0.0
bl_profile['vw'] = 0.0
col = [
       'y/delta_99', 'u+', 'u/u_e',	'v/u_e', 'w/u_e', 'r/r_e',
       'T/T_e',	'uu', 'vv',	'ww', 'uv',	'uw', 'vw'
       ]
bl_profile.to_csv(path + 'M2.1_ReD_55666_df.dat', sep='\t',
                  index=False, float_format='%1.8e', columns=col)
# %%
bl = pd.read_csv(path + 'Red9540.dat', sep=',', header=None,
                 index_col=False, skipinitialspace=True)
# %%
bl_profile = pd.read_csv(path + 'cZPGTBL_M2.00_Retau252.dat', sep='\t',
                         index_col=False, skipinitialspace=True)
# set path and column names
col = [
       'y/delta_99', 'u+', 'u/u_e',	'v/u_e', 'w/u_e', 'r/r_e',
       'T/T_e',	'uu', 'vv',	'ww', 'uv',	'uw', 'vw'
       ]
u_tau = bl_profile['u/u_e'].values[1:]/bl_profile['u+'].values[1:]
u_tau = np.mean(u_tau)

bl_profile['uu'] = (bl_profile['u_rms+'] * u_tau)**2
bl_profile['vv'] = (bl_profile['v_rms+'] * u_tau)**2
bl_profile['ww'] = (bl_profile['w_rms+'] * u_tau)**2
bl_profile['uv'] = bl_profile['uv+'] * u_tau * u_tau
bl_profile['uw'] = 0.0
bl_profile['vw'] = 0.0
bl_profile['w/u_e'] = 0.0

bl_profile.to_csv('cZPGTBL_M2.00_Retau252_orig.dat', sep='\t',
                  index=False, float_format='%1.8e', columns=col)

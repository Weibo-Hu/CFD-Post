# -*- coding: utf-8 -*-

import FlowVar as fv
import numpy as np

InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/8/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/Data/8/"
timezone = np.arange(880.00, 989.50 + 0.5, 0.5)
fv.ShockLoc(InFolder, OutFolder, timezone)

# -*- coding: utf-8 -*-

import FlowVar as fv
import numpy as np

InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/B/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Slice/Data/"
timezone = np.arange(600, 1000 + 0.5, 0.5)
fv.ShockLoc(InFolder, OutFolder, timezone)

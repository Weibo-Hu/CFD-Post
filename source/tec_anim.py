# -*- coding: utf-8 -*-
"""
Created on Wen May 11  22:39:40 2020
    This code uses pytecplot to plot 3d plots

@author: weibo
"""
# %% load libraries
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import tecplotlib as tl
import plt2pandas as p2p
import os
import sys
import numpy as np
import glob as glob

# %% instantaneous flow
path = "/media/weibo/IM2/FFS_M1.7ZA/"
p2p.create_folder(path)
pathF = path + 'Figures/'
pathin = path + "TP_data_01600750/"
dirs = glob.glob(pathin + '*.szplt')

tp.session.connect()
# tp.session.stop()
dataset = tp.data.load_tecplot_szl(dirs, read_data_option=2)
soltime = int(dataset.solution_times[0])
# with tp.session.suspend():
# tp.session.suspend_enter()
frame = tp.active_frame()
tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')

# frame setting
frame.width = 12.8
frame.height = 6.75
frame.position = (-1.0, 1.2)
tp.macro.execute_command('$!FrameLayout ShowBorder = No')

plot = frame.plot(PlotType.Cartesian3D)
plot.axes.orientation_axis.show=False
# value blank for FTB
#tl.show_ffs_wall(plot)
#blk_val = [-30, 10, 6]  
#axes_val = [-30, 10, 0, 6, -8, 8]
#axes = plot.axes
#tl.axis_set_ffs(axes), axes_val
# value blank for upstream FZA
blk_val = [-80, -45, 6]
axes_val = [-80, -45, 0, 6, -8, 8]
axes = plot.axes
tl.axis_set_ffs(axes, axes_val)
# value blank for downstream FZA
#blk_val = [-46, 5, 6]
#axes_val = [-46, 5, 0, 4, -8, 8]
#axes = plot.axes
#tl.axis_set_ffs(axes, axes_val)
plot.value_blanking.active=True
plot.value_blanking.constraint(0).variable = dataset.variable('x')
plot.value_blanking.constraint(0).comparison_operator=RelOp.LessThan
plot.value_blanking.constraint(0).comparison_value = blk_val[0]
plot.value_blanking.constraint(0).active=True
plot.value_blanking.constraint(1).variable = dataset.variable('x')
plot.value_blanking.constraint(1).comparison_operator=RelOp.GreaterThan
plot.value_blanking.constraint(1).comparison_value = blk_val[1]
plot.value_blanking.constraint(1).active=True
plot.value_blanking.constraint(2).variable = dataset.variable('y')
plot.value_blanking.constraint(2).comparison_operator=RelOp.GreaterThan
plot.value_blanking.constraint(2).comparison_value = blk_val[2]
plot.value_blanking.constraint(2).active=True
xpos = [53.5, 13.5]
ypos = [4.5, 65.6]
zpos = [8.5, 14]
tl.axis_lab(xpos, ypos, zpos)

# 3d view settings
view = plot.view
view.magnification = 1.0
# view.fit_to_nice()
view.rotation_origin = (10, 0.0, 0.0)
view.psi = 45
view.theta = 145
view.alpha = -140
view.position = (-200.5, 274, 331.5)
# view.distance = 300
view.width = 182 # 140 for TB; 205 # for ZA # 182 for upstream ZA; 
    
# limit values                                                                                                                          values
tl.limit_val(dataset, 'u')
tl.limit_val(dataset, 'p')

# create isosurfaces and its contour
var1 = 'L2-criterion'  # '<lambda_2>'
val1 = -0.1
var2 = 'u'
plot.show_isosurfaces = True
cont1 = plot.contour(0)
cont2 = plot.contour(1)
iso = plot.isosurface(0) 
val2 = np.round(np.linspace(-0.2, 1.0, 13), 2)
tl.plt_isosurfs(dataset, iso, cont1, var1, val1, cont2, var2, val2)

var3 = '|grad(rho)|'
val3 = np.linspace(0.0, 1.4, 29)
plot.show_slices = True
cont3 = plot.contour(3)
slc = plot.slice(0)
tl.plt_schlieren(dataset, slc, cont3, var3, val3, label=False)
# tp.session.suspend_exit()
tp.export.save_png(pathF + 'L2_ffs.png', width=2048)

# %% load data 
path = "/media/weibo/IM1/BFS_M1.7Tur/3D_DMD_1200/"
freq = "0p0755"
pathin = path + freq + "/"
pathout = path + freq + "_ani/"
file = '[' + freq + ']DMD'
figout  = 'p' + file
print(figout.replace(".", "p"))
dirs = os.listdir(pathin)
num = int(np.size(dirs)/2)
# tp.session.connect()
# num = 1
val1 = -0.3 # for u
val2 = -0.02  # for p
txtfl = open(pathout + 'levels.dat', "w")
txtfl.writelines('u` = ' + str(val1) + '\n')
txtfl.writelines('p` = ' + str(val2) + '\n')
txtfl.close()
for ii in range(num):
    ind = '{:03}'.format(ii)
    filelist = [file+ind+'A.plt', file+ind+'B.plt']
    print(filelist)
    datafile = [os.path.join(pathin, name) for name in filelist]
    dataset = tp.data.load_tecplot(datafile, read_data_option=2)
    SolTime = dataset.solution_times[0]

    # %% frame operation
    frame = tp.active_frame()
    # frame.load_stylesheet(path + 'video.sty')
    # turn off orange zone bounding box
    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
    # frame setting
    frame.width = 12.8
    frame.height = 7.5
    frame.position = (-1.0, 0.5)
    tp.macro.execute_command('$!FrameLayout ShowBorder = No')
    plot = frame.plot(PlotType.Cartesian3D)
    plot.axes.orientation_axis.show=False
    axes = plot.axes
    tl.axis_set(axes)
    xpos = [60, 12]
    ypos = [4.0, 74]
    zpos = [12.5, 18]
    tl.axis_lab(xpos, ypos, zpos)
    tl.axis_lab()

    # 3d view settings
    view = plot.view
    view.magnification = 1.0
    # view.fit_to_nice()
    view.rotation_origin = (10, 0.0, 0.0)
    view.psi = 45
    view.theta = 145
    view.alpha = -140
    view.position = (-46.5, 76, 94)
    # view.distance = 300
    view.width = 36.5
    
    # limit values                                                                                                                          values
    tl.limit_val(dataset, 'u`')
    tl.limit_val(dataset, 'p`')

    # create isosurfaces and its contour
    plot.show_isosurfaces = True
    cont = plot.contour(0)
    iso = plot.isosurface(0) 
    tl.plt_isosurf(dataset, iso, cont, 'p`', val2)

    # create slices and its contour
    plot.show_slices = False
    cont1 = plot.contour(5)
    slices = plot.slice(0)
    # tl.plt_slice(dataset, slices, cont1, 'p`', val2)

    # tl.figure_ind()   # show figure index
    # tl.show_time()  # show solution time
    tl.show_wall(plot)  # show the wall boundary

    # export figures
    outfile = pathout + figout + '{:02}'.format(int(SolTime))
    tp.export.save_png(outfile + '.png', width=2048)
    # tp.export.save_jpeg(outfile + '.jpeg', width=4096, quality=100) 
    
# %% generate animation
# %% Convert plots to animation
#import imageio
#from glob import glob
#import numpy as np
#from natsort import natsorted, ns
#path = "/media/weibo/IM1/BFS_M1.7Tur/3D_DMD_1200/"
#freq = "0p0755"
#pathout = path + freq + "_ani/"
#file = '[' + freq + ']DMD'
#dirs = glob(pathout + '[0p*.png')
#dirs = natsorted(dirs, key=lambda y: y.lower())
#flnm = path + file + '_Anima.mp4'
#with imageio.get_writer(flnm, mode='I', fps=12,
#                        macro_block_size=None) as writer:   
#    for ii in range(np.size(dirs)*6):
#        ind = ii % 32  # mod, get reminder
#        image = imageio.imread(dirs[ind])
#        writer.append_data(image)
#    writer.close()

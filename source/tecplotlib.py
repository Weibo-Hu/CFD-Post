# -*- coding: utf-8 -*-
"""
Created on Sun Jun 07  22:39:40 2020
    This code uses pytecplot to plot 3d plots

@author: weibo
"""
# %% load libraries
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
from glob import glob
import numpy as np

def limit_val(dataset, var):
    minv = dataset.variable(var).min()
    maxv = dataset.variable(var).max()
    print("Limit value of " + var, minv, maxv)

def plt_isosurf(dataset, iso, cont, var, val):
    # create contour for isosurfaces for velocity
    # create isosurfaces
    iso.show = True
    iso.contour.flood_contour_group = cont
    iso.definition_contour_group.variable = dataset.variable(var)
    iso.isosurface_selection = IsoSurfaceSelection.TwoSpecificValues
    iso.isosurface_values = (val, -val)
    iso.effects.lighting_effect = LightingEffect.Gouraud
    iso.contour.show = True
    iso.contour.use_lighting_effect = True
    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 20

    cont.variable = dataset.variable(var)
    cont.levels.reset_levels(np.linspace(val/2*3, -val/2*3, 7))
    cont.colormap_name = 'Diverging - Blue/Red'    
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont.colormap_filter.continuous_min = val/2*3
    cont.colormap_filter.continuous_max = -val/2*3
    cont.legend.show = False


def plt_slice(dataset, slc, cont, var, val, label=True):
    slc.show = True
    slc.orientation = SliceSurface.ZPlanes
    slc.origin = (slc.origin[0], slc.origin[1], -8)
    slc.contour.flood_contour_group = cont
    slc.contour.show = True

    cont.variable = dataset.variable(var)
    cont.levels.reset_levels(np.linspace(val/2*3, -val/2*3, 7))
    cont.colormap_name = 'Diverging - Purple/Green'
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont.colormap_filter.continuous_min = val/2*3
    cont.colormap_filter.continuous_max = -val/2*3
    cont.legend.show = True
    cont.legend.vertical = False
    cont.legend.number_font.size=2.8
    cont.legend.row_spacing=1.3
    cont.legend.number_font.typeface='Times New Roman'
    cont.legend.show_header=False
    cont.legend.box.box_type=tp.constant.TextBox.None_
    cont.labels.step=2
    cont.legend.position=(30, 91)
    
    if label == True:
        tp.macro.execute_command("""$!AttachText 
            AnchorPos
              {
              X = 29
              Y = 86.5
              }
            TextShape
              {
              SizeUnits = Frame
              Height = 4.2
              }
            TextType = LaTeX
            Text = '$p^{\\prime}$'""")


def axis_set(axes_obj):
    axes_obj.grid_area.filled=False
    axes_obj.x_axis.show=True
    axes_obj.x_axis.min=-5
    axes_obj.x_axis.max=20
    axes_obj.x_axis.tick_labels.font.size=3.0
    axes_obj.x_axis.tick_labels.font.typeface='Times'
    axes_obj.x_axis.title.show=False

    axes_obj.y_axis.show=True
    axes_obj.y_axis.min=-3
    axes_obj.y_axis.max=2
    axes_obj.y_axis.tick_labels.font.size=3.0
    axes_obj.y_axis.tick_labels.font.typeface='Times'
    axes_obj.y_axis.title.show=False
    axes_obj.y_axis.ticks.show_on_opposite_edge=True
    axes_obj.y_axis.tick_labels.show_on_opposite_edge=True
    axes_obj.y_axis.ticks.show=False
    axes_obj.y_axis.tick_labels.show=False
    
    axes_obj.z_axis.show=True
    axes_obj.z_axis.min=-8
    axes_obj.z_axis.max=8
    axes_obj.z_axis.tick_labels.font.size=3.0
    axes_obj.z_axis.tick_labels.font.typeface='Times'
    axes_obj.z_axis.title.show=False
    axes_obj.z_axis.ticks.auto_spacing=False
    axes_obj.z_axis.ticks.spacing=4
    #axes_obj.x_axis.title.show_on_opposite_edge=True
    #axes_obj.x_axis.title.show_on_opposite_edge=False


def figure_ind():
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 5.0
          Y = 92
          }
        TextShape
          {
          FontFamily = 'Times New Roman'
          IsBold = No
          SizeUnits = Frame
          Height = 3.6
          }
        Text = '(a)'""")

def axis_lab(xpos, ypos, zpos):
    str_x = """$!AttachText 
        AnchorPos
          {
          X = 
          Y = 
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.5
          }
        TextType = LaTeX
        Text = '$x/\\delta_0$'"""
    str_x = str_x.replace("X = \n", "X = "+str(xpos[0])+"\n")
    str_x = str_x.replace("Y = \n", "Y = "+str(xpos[1])+"\n")
    tp.macro.execute_command(str_x)
    
    str_y = """$!AttachText 
        AnchorPos
          {
          X = 
          Y = 
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.5
          }
        TextType = LaTeX
        Text = '$y/\\delta_0$'"""
    str_y = str_y.replace("X = \n", "X = "+str(ypos[0])+"\n")
    str_y = str_y.replace("Y = \n", "Y = "+str(ypos[1])+"\n")
    tp.macro.execute_command(str_y)
    
    str_z = """$!AttachText 
        AnchorPos
          {
          X = 
          Y = 
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.5
          }
        TextType = LaTeX
        Text = '$z/\\delta_0$'"""
    str_z = str_z.replace("X = \n", "X = "+str(zpos[0])+"\n")
    str_z = str_z.replace("Y = \n", "Y = "+str(zpos[1])+"\n")
    tp.macro.execute_command(str_z)

def show_time():
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 80.5
          Y = 90
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 3.6
          }
        TextType = LaTeX
        Text = '$\\theta=\\frac{\\pi}{16}\\times$'""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 88.0
          Y = 91.2
          }
        TextShape
          {
          FontFamily = 'Times New Roman'
          IsBold = No
          SizeUnits = Frame
          Height = 3.0
          }
        Text = '&(solutiontime)'""")

def show_wall(plot):
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = -3
        Z1 = -8
        X2 = 20
        Y2 = -3
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = -3
        Z1 = -8
        X2 = 0
        Y2 = 0
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = -5
        Y1 = 0
        Z1 = -8
        X2 = 0
        Y2 = 0
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    plot.use_lighting_effect=True
    plot.show_shade=True
    plot.fieldmap(-1).shade.show=True
    plot.fieldmap(-2).shade.show=True
    plot.fieldmap(-3).shade.show=True
    plot.fieldmap(-1).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(-2).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(-3).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces

def plt_isosurfs(dataset, iso, cont1, var1, val1,
                 cont2, var2, val2, label=True):
    # create contour for isosurfaces for velocity
    # create isosurfaces
    iso.show = True
    
    cont1.variable = dataset.variable(var1)
    iso.definition_contour_group = cont1
    iso.isosurface_selection = IsoSurfaceSelection.OneSpecificValue
    iso.isosurface_values = (val1, -val1)
    iso.effects.lighting_effect = LightingEffect.Gouraud
    iso.contour.show = True
    iso.contour.use_lighting_effect = True
    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 20

    iso.contour.flood_contour_group = cont2    
    cont2.variable = dataset.variable(var2)
    cont2.levels.reset_levels(val2)
    cont2.colormap_name = 'Small Rainbow'    
    cont2.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont2.colormap_filter.continuous_min = np.min(val2)
    cont2.colormap_filter.continuous_max = np.max(val2)   
    cont2.legend.show = True
    cont2.legend.vertical = False
    cont2.legend.number_font.size=3.2
    cont2.legend.row_spacing=1.6
    cont2.legend.overlay_bar_grid=False
    cont2.legend.number_font.typeface='Times New Roman'
    cont2.legend.show_header=False
    cont2.legend.box.box_type=tp.constant.TextBox.None_
    cont2.labels.step=3
    cont2.legend.position=(27, 88)
    if label == True:
        tp.macro.execute_command("""$!AttachText 
            AnchorPos
              {
              X = 26
              Y = 82.5
              }
            TextShape
              {
              SizeUnits = Frame
              Height = 4.2
              }
            TextType = LaTeX
            Text = '$u/u_{\\infty}$'""")


def plt_schlieren(dataset, slc, cont, var, val,
                  label=True, continuous=False):
    slc.show = True
    slc.orientation = SliceSurface.ZPlanes
    slc.origin = (slc.origin[0], slc.origin[1], -8)
    slc.contour.flood_contour_group = cont
    slc.contour.show = True

    cont.variable = dataset.variable(var)
    cont.levels.reset_levels(val)
    cont.colormap_name='GrayScale'
    cont.colormap_filter.reversed=True
    cont.colormap_filter.distribution = ColorMapDistribution.Banded
    if continuous == True:
      cont.colormap_filter.distribution = ColorMapDistribution.Continuous
      cont.colormap_filter.continuous_min = np.min(val)
      cont.colormap_filter.continuous_max = np.max(val)
    else:
      cont.colormap_filter.distribution = ColorMapDistribution.Banded
    cont.legend.show = False
    cont.legend.vertical = False
    cont.legend.number_font.size=2.8
    cont.legend.row_spacing=1.3
    cont.legend.number_font.typeface='Times New Roman'
    cont.legend.show_header=False
    cont.legend.box.box_type=tp.constant.TextBox.None_
    cont.labels.step=2
    cont.legend.position=(30, 91)
    
    if label == True:
        tp.macro.execute_command("""$!AttachText 
            AnchorPos
              {
              X = 29
              Y = 86.5
              }
            TextShape
              {
              SizeUnits = Frame
              Height = 4.2
              }
            TextType = LaTeX
            Text = '$p^{\\prime}$'""")

def axis_set_ffs(axes_obj, axes_val):
    axes_obj.grid_area.filled=False
    axes_obj.x_axis.show=True
    axes_obj.x_axis.min=axes_val[0]
    axes_obj.x_axis.max=axes_val[1]
    axes_obj.x_axis.tick_labels.font.size=2.3
    axes_obj.x_axis.tick_labels.font.typeface='Times New Roman'
    axes_obj.x_axis.title.show=False
    
    axes_obj.preserve_scale=True
    axes_obj.y_axis.show=True
    axes_obj.y_axis.min=axes_val[2]
    axes_obj.y_axis.max=axes_val[3]
    axes_obj.y_axis.tick_labels.font.size=2.3
    axes_obj.y_axis.tick_labels.font.typeface='Times New Roman'
    axes_obj.y_axis.title.show=False
    axes_obj.y_axis.ticks.show_on_opposite_edge=True
    axes_obj.y_axis.tick_labels.show_on_opposite_edge=True
    axes_obj.y_axis.tick_labels.angle=20
    axes_obj.y_axis.ticks.show=False
    axes_obj.y_axis.tick_labels.show=False
    
    axes_obj.z_axis.show=True
    axes_obj.z_axis.min=axes_val[4]-0.1
    axes_obj.z_axis.max=axes_val[5]
    axes_obj.z_axis.tick_labels.font.size=2.3
    axes_obj.z_axis.tick_labels.font.typeface='Times New Roman'
    axes_obj.y_axis.tick_labels.angle=20
    axes_obj.z_axis.title.show=False
    axes_obj.z_axis.ticks.auto_spacing=False
    axes_obj.z_axis.ticks.spacing=4
    #axes_obj.x_axis.title.show_on_opposite_edge=True
    #axes_obj.x_axis.title.show_on_opposite_edge=False

def show_ffs_wall(plot):
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 100
        JMax = 10
        KMax = 10
        X1 = -60
        Y1 = 0
        Z1 = -8
        X2 = 0
        Y2 = 0
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = 0
        Z1 = -8
        X2 = 0
        Y2 = 3
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = 3
        Z1 = -8
        X2 = 20
        Y2 = 3
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    plot.use_lighting_effect=True
    plot.show_shade=True
    plot.fieldmap(-1).shade.show=True
    plot.fieldmap(-2).shade.show=True
    plot.fieldmap(-3).shade.show=True
    plot.fieldmap(-1).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(-2).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(-3).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces


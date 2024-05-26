# %% generate curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
x1 = np.linspace(-1, 26, 660, endpoint=True)
x2 = np.linspace(26, 108.811328, 2000, endpoint=True)
x3 = np.linspace(108.811328, 361, 6000, endpoint=True)
y1 = np.zeros(np.size(x1))
alpha = 0.758735
lambd = 2*np.pi/alpha
amplit = 0.1* 2.6
y2 = amplit*np.sin(alpha*(x2-26)+np.pi/2) - amplit
y3 = np.zeros(np.size(x3))

xx = np.concatenate((x1[:-1], x2, x3[1:]))
yy = np.concatenate((y1[:-1], y2, y3[1:]))
spoint = [-1, -2, 0]
epoint = [361, -2, 0]

xy = np.vstack((xx, yy))
df = pd.DataFrame(data=np.transpose(xy), columns=['x', 'y'])
df.to_csv('wavy.dat', sep=',', index=False, float_format='%9.6f')
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(xx, yy, 'b-')
ax.set_xlim((np.min(xx), np.max(xx)))
ax.grid(which='both')
plt.savefig('wavy_wall.svg')
plt.show()


# %% load module
import sys
# sys.path.append('C:/APP/FreeCAD-0.19/bin/')
sys.path.append("/usr/lib/freecad-python3/lib/")
import FreeCAD
import Mesh
import Draft
import Part

# %% method 1: create center line of the surface
doc = FreeCAD.newDocument()
# doc.saveAs(u"D:/cases/sin_wall.FCStd")
finm = 'wavy8'
doc.saveAs("/mnt/share/cases/" + finm + ".FCStd")
# App.getDocument("Unnamed").saveAs(u"D:/cases/low_Re_case/sin_wall.FCStd")
pl =FreeCAD.Placement()
points = []
for i in range(np.size(xx)):
    points.append(FreeCAD.Vector(xx[i], yy[i], 0.0))
    # App.getDocument('Unnamed').getObject('Sketch').addGeometry(Part.Point(FreeCAD.Vector(xx[i], yy[i], 0.0)))
# line = doc.addObject("Part::Polygon", "Polygon")
line = Draft.makeWire(points, placement=pl, closed=False, face=False, support=None)
line.Label="wavy"
Draft.autogroup(line)
FreeCAD.ActiveDocument.recompute()
# % extrude to surface
FreeCAD.ActiveDocument.addObject('Part::Extrusion','Extrude')
f = FreeCAD.ActiveDocument.getObject('Extrude')
f.Base = line
f.DirMode = "Custom"
f.Dir = FreeCAD.Vector(0.00000000, 0.00000000, 1.00000000)
f.DirLink = None
f.LengthFwd = 8.60000000000000
f.LengthRev = 8.60000000000000
f.Solid = False
f.Reversed = False
f.Symmetric = False
f.TaperAngle = 0.000000000000000
FreeCAD.ActiveDocument.recompute()
# doc.saveAs(u"D:/cases/sin_wall.FCStd")
doc.saveAs("/mnt/share/cases/" + finm + ".FCStd")

# % extrude to cube
FreeCAD.ActiveDocument.addObject('Part::Extrusion','Extrude001')
c = FreeCAD.ActiveDocument.getObject('Extrude001')
c.Base = f
c.DirMode = "Custom"
c.Dir = FreeCAD.Vector(0.00000000, -1.000000000, 0.000000000)
c.DirLink = None
c.LengthFwd = 4.000000000000000
c.LengthRev = 0.000000000000000
c.Solid = False
c.Reversed = False
c.Symmetric = False
c.TaperAngle = 0.000000000000000
FreeCAD.ActiveDocument.recompute()
# doc.saveAs(u"D:/cases/sin_wall.FCStd")
doc.saveAs("/mnt/share/cases/" + finm + ".FCStd")
__objs__=[]
__objs__.append(FreeCAD.ActiveDocument.getObject("Extrude"))
# Mesh.export(__objs__,u"D:/cases/sin_wall.stl")
Mesh.export(__objs__,"/mnt/share/cases/" + finm + ".stl")
del __objs__

# %% a better method 2: create center line of the surface
doc = FreeCAD.newDocument()
doc.saveAs(u"D:/cases/wavy_9016.FCStd")
finm = 'wavy_0916'
# create points
points = [FreeCAD.Vector(spoint)]
for i in range(np.size(xx)):
    points.append(FreeCAD.Vector(xx[i], yy[i], 0.0))
points.append(FreeCAD.Vector(epoint))
points.append(FreeCAD.Vector(spoint))
# generate line
line = Part.makePolygon(points)
Part.show(line)
sline = Part.Wire(line.Edges)
FreeCAD.ActiveDocument.recompute()
# generate surface
surf = Part.makeFace(sline, 'Part::FaceMakerSimple')
Part.show(surf)
# select surface object
FreeCAD.Gui.Selection.addSelection('Unnamed', 'Shape001')
# generate 3d object
# FreeCAD.Gui.runCommand('Part_Extrude', 0)
FreeCAD.ActiveDocument.addObject('Part::Extrusion','Extrude')
f = FreeCAD.ActiveDocument.getObject('Extrude')
f.Base = FreeCAD.ActiveDocument.getObject('Shape001')
f.DirMode = "Custom"
f.Dir = FreeCAD.Vector(0.00000000, 0.00000000, 1.00000000)
f.DirLink = None
f.LengthFwd = 8.600000000000000
f.LengthRev = 8.600000000000000
f.Solid = False
f.Reversed = False
f.Symmetric = False
f.TaperAngle = 0.000000000000000
FreeCAD.ActiveDocument.recompute()
FreeCAD.Gui.Selection.addSelection('Unnamed', 'Extrude')


# line1 = Part.makeLine((-1,0,0),(-1,-1,0))
# Part.show(line1)

# surf = sline.extrude(FreeCAD.Vector(0, 0, 4))
# %% covert to ascii
path = '/mnt/data3/wavy_0916/'
finm = 'wavy_0916'
file_bin = path + finm + '.stl '
file_asc = path + finm + '_ascii.stl'
os.system('stl2ascii '+ file_bin + file_asc)
# %%
path = '/home/weibo/'
finm = 'test-Fusion'
file_bin = path + finm + '.stl '
file_asc = path + finm + '_ascii.stl'
os.system('stl2ascii '+ file_bin + file_asc)
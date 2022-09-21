# coding: utf-8
# pylint: disable=invalid-name
""" demo for distmesh 3D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function




import numpy as np
from matplotlib import pyplot as plt

from pyeit.mesh import distmesh
from pyeit.mesh.utils import dist, edge_project
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pyeit.mesh.quality as quality
import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot
from pyeit.mesh.shape import thorax, area_uniform, rectangle, ball, ellipse, rectangle0, box_circle, cylinder, circle, thorax2D

# tetrahedron meshing in a 3D bbox
bbox = [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]
num = 16

el_pos = np.arange(num)
p_fix = [
        (-5.6884, -113.73, 0.65624),
        (59.191, -109.53, 0.65618),
        (116.46, -87.213, 0.65613),
        (156.06, -38.888, 0.65632),
        (188.86, 15.47, 0.6566),
        (188.9, 82.114, 0.65688),
        (135.82, 128.86, 0.65688),
        (72.871, 157.22, 0.65678),
        (4.3429, 161.59, 0.65662),
        (-64.856, 149.68, 0.65635),
        (-119.59, 117.78, 0.65621),
        (-162.99, 78.197, 0.65636),
        (-178.72, 24.152, 0.6565),
        (-152.17, -33.617, 0.65658),
        (-118.05, -80.938, 0.65635),
        (-68.639, -109.89, 0.65628)
    ]
p_fix = np.array(p_fix)
       # 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
ms, el_pos = mesh.create(n_el=num, h0=0.15, bbox=bbox, fd=ball)
# print mesh quality
p = ms["node"]
t = ms["element"]
print("points =", p.shape)
print("simplices =", t.shape)
# plot
mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
# create random color
f = np.random.randn(p.shape[0])
mplot.tetplot(p, t, f, alpha=0.25)

    # build triangles

# coding: utf-8
# pylint: disable=invalid-name

# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""


import numpy as np

import pyeit.mesh as mesh
import pyeit.mesh.plot as mplot

# tetrahedron meshing in a 3D bbox
bbox = [[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]
# 3D Mesh shape is specified with fd parameter in the instantiation, e.g : fd=ball , Default in 3D :fd=ball
ms = mesh.create(h0=0.15, bbox=bbox)

# print mesh quality
p = ms.node
t = ms.element
ms.print_stats()

p = ms["node"]
t = ms["element"]
print("points =", p.shape)
print("simplices =", t.shape)
# plot
mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
# create random color
# f = np.random.randn(p.shape[0])
"""
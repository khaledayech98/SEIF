import csv

import matplotlib.pyplot as plt
import seaborn as sns
from dask import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



from statistics import mean, median, fmean

import warnings
import os
import pyvista as pv
import tetgen
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt, axes
from numpy.lib import savetxt
from tetgen import TetGen

from pyeit.eit import jac, svd
from vispy.plot import fig
from scipy.stats import skew


import pyeit.mesh.plot as mplot
from build.lib.pyeit.eit import greit
from build.lib.pyeit.mesh import shape

from pyeit.mesh.wrapper import set_perm
from pyeit.eit.utils import eit_scan_lines
import pandas as pd
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts




pv.set_plot_theme('document')
mesh = pv.read(r'D:\Desktop\EIT\thorax minus minus.stl')
#cpos = mesh.plot()

#print("will reduce now.........")
#tetreduced=mesh.decimate_pro(0.5)
#print("done reducing.........")

tet = tetgen.TetGen(mesh)
tetraMesh=tet.tetrahedralize(order=1, mindihedral=10, minratio=1.1)
print(type(tetraMesh))

vertices=tetraMesh[0]
#int_vertices= vertices.astype(int)
print("vertices:", vertices)
tetras=tetraMesh[1]
print("tetras:", tetras)




#nombre des electrodes
#skeweness + mesh evaluation ( mathematical )
#print(skew(vertices, bias=False))
#el_pos = np.array([1, 4, 6, 8, 11, 14, 16, 19, 22, 24, 26, 28, 30, 32, 34, 37, 39, 42, 44, 46, 48, 51, 53, 56, 58, 60, 62, 64, 66, 68, 71, 74])
el_pos = np.array([1, 4, 6, 8, 11, 14, 16, 19, 22, 24, 26, 28, 30, 32, 34, 37])

#el_pos = np.array([1, 6, 11, 16,  22, 26, 30, 34])
num =   16
p_fix = shape.fix_points_circle(ppl=num)

# firs num nodes are the positions for electrodes
el_pos = np.arange(num)
# here we define the initial permitivity (5000)
tri_perm = np.array(5000. * np.ones(len(tetras)))


# this is the output of the create funtion
mesh_obj = {
    "element": tetras,
    "node": vertices,
    "perm": tri_perm,
}
# Creating the dataframe
#obj_df = pd.read_csv("mesh_obj", header=None, delimiter=' ')

# skip the na values
# find skewness in each row
#obj_df.skew(axis=1, skipna=True)
grid = tet.grid
#grid.plot(show_edges=True)
cqual = mesh.compute_cell_quality('min_angle')
print(cqual)


#mesh.plot_vp_vs_profile(depth=True)
#cqual.plot(show_edges=True)
plotter = pv.Plotter(shape=(1, 2))

plotter.add_text("Mesh", font_size=30)
plotter.add_mesh(grid, show_edges= True)
plotter.subplot(0,1)
plotter.add_text("Mesh quality", font_size=30)
plotter.add_mesh(cqual, show_edges= True)
plotter.show()
qual = mesh.compute_cell_quality(quality_measure='scaled_jacobian')
print(qual)
cpos = [
    (10.10963531890468, 4.61130688407898, -4.503884867626516),
    (1.2896420468715433, -0.055387528972708225, 1.1228250502811408),
    (-0.2970769821136617, 0.9100381451936025, 0.2890948650371137),
]
qual.plot(cpos=cpos, scalars='CellQuality')
# extract node, element, alpha
pts = mesh_obj["node"]
tri = mesh_obj["element"]
tri_perm0 = mesh_obj["perm"]


x, y = pts[:, 0], pts[:, 1]
print("will plot now........")
""" Plot the mesh + electrodes + triangle numbers """
# plot mesh

mplot.tetplot(pts, tri, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
print("done........")
#print("tri", tri)
#print("pts: ", pts)
# calculate simulated data
print("will start calculating forward")
#fwd = Forward(mesh_obj, el_pos)
print("done calculating forward")
print ("vertices:", len(vertices))
print("tetras:", len(tetras))
anomaly = {'name': 'ball',"x": 0, "y": 0, "z": 0, "d": 50, 'perm': 600}
n_anom = 1
anoms = np.empty((n_anom,), dtype='O')
anoms[0] = anomaly
mesh_new = set_perm(mesh_obj, anomaly=anoms, background= anomaly["perm"])
tri_perm = mesh_new["perm"]
""" 1. FEM forward simulations """
# setup EIT scan conditions
el_dist, step = 1, 1 #7(opposite)
ex_mat = eit_scan_lines(16, el_dist)
fwd = Forward(mesh_obj, el_pos)
f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])
# in python, index start from 0
#ex_line = ex_mat[1].ravel()
#ex_line = ex_mat[2].ravel()
# change alpha

#node_ds = sim2pts(pts,tri, tri_perm)
#print(len(node_ds))
print(tri_perm)

# solving once using fem
#f1, _ = fwd.solve(ex_line, perm=tri_perm)
#f1 = np.real(f1)
voltage = f1.v
with open('voltage3.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(['voltage number','voltage amplitude','permittivity'])
    for i in range(len(voltage)):
     writer.writerow([(i), (voltage[i]*(10**6)), ((mesh_new["perm"][i]))])
print(len(f1))
print("f1=", f1)
plt.plot(voltage)
plt.xlabel('Number of the resulted boundary voltages', fontsize=24)
plt.ylabel('voltage amplitude', fontsize= 24)
plt.show()




# in python, index start from 0


#ex_line = ex_mat[2].ravel()
# define an anomaly to add to the mesh
#anomaly1 = {'name': 'ellipse', "x": 30, "y": 10, "a": 30, "b": 50, "angle": 60, "perm": 300.0}
#anomaly = {'name': 'ball',"x": 0, "y": 0, "z": 0, "d": 50, 'perm': 500}
#anomaly2 = {'name': 'ball',"x": 100, "y": -100, "z": 0, "d": 50, "perm": 700.0}
#anomaly3 = {'name': 'ball',"x": 50, "y": 50, "z": 50, "d": 50, "perm": 1000.0}
#anomaly2 = {'name': 'ellipse', "x": 100, "y": 200, "a": 50, "b": 50, "angle": 45, "perm": 300.0}

#anoms[1] = anomaly2
#anoms[2] = anomaly3

#anomaly1 = pv.read(r'D:\Desktop\EIT\lungs.stl')

#anomaly1 = {'name': 'ball',"x": 0, "y": 0, "z": 0, "d": 50, "perm": 500.0}

#mesh_new = set_perm(mesh_obj, anomaly=anoms, background=None)
#perm= mesh_new["perm"]





# save to csv file
#savetxt('data2.csv', (f1* pow(10, 6)), delimiter=',')



# for i in tri:
# print(type(i))
#   np.append(i,[1])


#print("shape pts:", pts.shape, "shape tri:", tri.shape, "shape ds:", tri_perm0.shape)

#print("node_ds", node_ds)


#tgen.write('grid.vtk', binary=False)

#print("I am rendering now.....")
#mplot.tetplot(pts, tri, vertex_color=node_ds, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.1)
'''
eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
eit.setup(p=0.25, lamb=1.0, method="lm")
# lamb = lamb * lamb_decay
ds = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)
node_ds = sim2pts(pts, tri, np.real(ds))


# mplot.tetplot(p, t, edge_color=(0.2, 0.2, 1.0, 1.0), alpha=0.01)
mplot.tetplot(pts, tri, vertex_color=node_ds, alpha=1.0)
'''
'''
"""  JAC solver """
eit = svd.SVD(mesh_obj, el_pos, ex_mat=ex_mat, step=step, perm=1.0, parser="std")
eit.setup(n=35, method="svd")
ds = eit.solve(f1.v, f0.v, normalize=True)
ds_n = sim2pts(pts, tri, np.real(ds))

# plot ground truth
fig, ax = plt.subplots(figsize=(6, 4))
delta_perm = mesh_new["perm"] - mesh_obj["perm"]
im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
fig.colorbar(im)
ax.set_aspect("equal")

# plot EIT reconstruction
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
for i, e in enumerate(el_pos):
    ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
fig.colorbar(im)
ax.set_aspect("equal")
# fig.set_size_inches(6, 4)
# plt.savefig('../figs/demo_jac.png', dpi=96)
plt.show()
'''


import numpy

import plotly.graph_objects as go
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from pyeit.mesh import multi_shell, multi_circle


# Using an existing stl file:
your_mesh = mesh.Mesh.from_file('model_remeshed_with_16electrodes.stl')

# Or creating a new mesh (make sure not to overwrite the `mesh` import by
# naming it `mesh`):
VERTICE_COUNT = 100
data = numpy.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
your_mesh = mesh.Mesh(data, remove_empty_areas=False)

# The mesh normals (calculated automatically)
your_mesh.normals
# The mesh vectors
your_mesh.v0, your_mesh.v1, your_mesh.v2
# Accessing individual points (concatenation of v0, v1 and v2 in triplets)
assert (your_mesh.points[0][0:3] == your_mesh.v0[0]).all()
assert (your_mesh.points[0][3:6] == your_mesh.v1[0]).all()
assert (your_mesh.points[0][6:9] == your_mesh.v2[0]).all()
assert (your_mesh.points[1][0:3] == your_mesh.v0[1]).all()
#n_fan = 6
#n_layer = 12
#r_layers = [n_layer - 1]
#perm_layers = [0.01]
#mesh_obj, el_pos = multi_shell(
 #   n_fan=n_fan, n_layer=n_layer, r_layer=r_layers, perm_per_layer=perm_layers
#)
your_mesh.save('C:\home\pyEIT-master\examples\new_file.stl')


# Create a new plot
fig = pyplot.figure()
axes = mplot3d.Axes3D(fig)
your_mesh = mesh.Mesh.from_file('C:\home\pyEIT-master\examples/model_remeshed_with_16electrodes.stl')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()

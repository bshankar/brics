from dolfin import *
from sys import argv

# a simple script to quickly visualize a mesh

comm = mpi_comm_world()
f = HDF5File(comm, argv[1], 'r')
mesh = Mesh()
f.read(mesh, 'mesh', True)
plot(mesh)
interactive()

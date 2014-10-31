"""
Brics is a finite element solver for Rayleigh Benard equations
based on FEniCS and PETSc.
Copyright (C) 2013-2014 E. Bhavani Shankar

Brics is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Brics is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


from dolfin import *
import numpy as np

# tools to generate and save meshes

def is_negative(x):
    """
    Generates the missing sign while stretching a cylinder's coordinates
    """
    if x < 0:
        return -1
    return 1

def denser(x, s=1.3):
    """
    input:
    x --- equally spaced np.array [-1, ... , 1]
    s --- stretch factor 
    
    output:
    x' = [-1, ..., 1] with elements stretched towards -1 and 1
    """
    if x < 0:
        return -np.abs(x)**(1/s)
    return x**(1/s)

denser = np.vectorize(denser)
is_negative = np.vectorize(is_negative)

def saveMesh(mesh, filename, outputFormat="hdf5"):
    """
    save a mesh in given output format
    
    Output formats:
    1. xml --- calculations with dolfin
    2. pvd --- 3d visualization of mesh
    3. hdf5 --- saving and loading data
    """
    if outputFormat == 'xml':
        if MPI.process_number() == 0:
            print "saving mesh as xml.gz ..."
        file = File(filename + '.xml.gz', "compressed")
        file << mesh
    elif  outputFormat == 'pvd':
        if MPI.process_number() == 0:
            print "saving mesh as pvd ..."
        file = File(filename + '.pvd')
        file << mesh
        
    elif outputFormat == 'hdf5':
        if MPI.process_number() == 0:
            print "saving mesh in hdf5 format ..."
        comm = mpi_comm_world()
        f = HDF5File(comm, filename + '.h5', 'w')
        f.write(mesh, 'mesh')

    elif outputFormat == 'xdmf':
        if MPI.process_number() == 0:
            print "saving mesh as an xdmf"
        file = File(filename + ".xdmf")
        file << mesh

    else:
        print "Unknown format! mesh could not be saved."
     

def box_mesh_name(dim, res):
    # give a consistent name to a box mesh
    mesh_id  = "".join(str(dim[i]) + "_" for i in range(len(dim)))
    mesh_id += "".join(str(res[i]) + "_" for i in range(len(res)))
    return ".cache/" + mesh_id


def DenserBox(dim, res, s=1.2, outputFormat="hdf5"):    
    
    if len(dim) == 2:
        mesh = RectangleMesh(0, 0, dim[0], dim[1], res[0], res[1])
    else:
        mesh = BoxMesh(0, 0, 0, dim[0], dim[1], dim[2], res[0], res[1], res[2])

    # extract the mesh coordinates
    # re-scale and displace them to go from -1 to 1
    X = 2*mesh.coordinates()[:, 0]/dim[0] - 1
    Y = 2*mesh.coordinates()[:, 1]/dim[1] - 1
    Z = None
    if len(dim) == 3:
        Z = 2*mesh.coordinates()[:, 2]/dim[2] - 1
        
    #make them denser towards ends 
    # re-scale and displace to from 0 to a or 0 to b
    #mesh.coordinates()[:, 0] = 0.5*(denser(X, s) + 1)*dim[0]
    #mesh.coordinates()[:, 1] = 0.5*(denser(Y, s) + 1)*dim[1]
    if len(dim) == 3:
        mesh.coordinates()[:, 2] = 0.5*(denser(Z, s) + 1)*dim[2]
    # save the mesh
    saveMesh(mesh, box_mesh_name(dim, res), outputFormat) 
        

def DenserEllipticCylinder(h, Nv, Nz, s=1.3, outputFormat='xml'):
    # A rectangle with denser regions at z = 0, z = a and at v = 0, pi

    mesh = RectangleMesh(0, 0, 2*pi,  h, Nv, Nz)

    # extract the mesh coordinates
    # re-scale and displace them to go from -1 to 1
    V = 2*mesh.coordinates()[:, 0]
    Z = 2*mesh.coordinates()[:, 1]/h - 1

    #make them denser towards ends 
    # re-scale and displace to from 0 to a or 0 to b
    mesh.coordinates()[:, 1] = 0.5*(denser(Z, s) + 1)*h
    saveMesh(mesh, 'EllipticCylinder%g_%g_%g_%g'%(h, Nv, Nz, s), outputFormat)

    
def DenserCylinder(radius, height, m, n, s=1.3, outputFormat='xml'):
    """
    Generate a cylindrical mesh and save it. stretched radially outward
    and towards towards its caps. The axis of the cylinder is aligned along z axis 
    and the center of its base is at the origin.
    
    Input:
    radius --- radius of the cylinder
    height --- height of the cylinder
    m      --- approximate the curved surface of cylinder with m-polygon
    n      --- subdivide the cylinder into n cubes
    s      --- stretch factor (= 1.3 by default)
    
    Output formats:
    1. xml --- calculations with dolfin
    2. pvd --- 3d visualization of mesh
    """
    
    cylinder = Cylinder(Point(0, 0, height), Point(0, 0, 0), radius, m)
    mesh = Mesh(cylinder, n)
    
    # extract all the coordinates
    X = mesh.coordinates()[:, 0]
    Y = mesh.coordinates()[:, 1]
    Z = mesh.coordinates()[:, 2]
        
    Z      = np.array(Z)
    offset = np.max(Z)/2
    Z      = Z - offset
    # stretch towards caps
    mesh.coordinates()[:, 2] = denser(Z, s) + offset

    X_neg = is_negative(X)
    Y_neg = is_negative(Y)

    R     = np.sqrt(X**2 + Y**2)
    Theta = np.arctan(Y/X)
    # stretch radially outwards
    R = denser(R, s)
    mesh.coordinates()[:, 0] = X_neg*R*np.cos(Theta)
    mesh.coordinates()[:, 1] = X_neg*R*np.sin(Theta)
    
    saveMesh(mesh, 'Cylinder%g_%g_%g_%g_%g'%(height, radius, m, n, s), outputFormat)


def interpolate_to_mesh(u1, mesh1, mesh2, orders=(1,1)):
    """
    u1    ---- dolfin function
    mesh1 ---- From mesh
    mesh2 ---- to mesh
    order ---- order of polynomials on from Func space to to Func space.
    
    Returns:
        u2 ---- dolfin function
    """

    dim = u1.rank() + 1

    if dim == 1:
        V2 = FunctionSpace(mesh2, 'CG', orders[1])
    else:
        V2 = VectorFunctionSpace(mesh2, 'CG', orders[1], dim=dim)

    u2 = interpolate(u1, V2)
    return u2
    
    

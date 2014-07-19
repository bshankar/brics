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
from mesh_tools import *
import os.path


class box:
    """
    Generate/load an appropriate box mesh
    Create Function spaces and functions
    specify the boundary conditions
    """
    
    def __init__(self, dim, res, orders=(1,1,1), scales=('small', 'small'), pb=None):
        """
		dim    ----  dimensions of the box
        res    ----  resolution of the box
        orders ----  order of basis function (higher orders are more accurate but slower)
        pb     ----  periodic boundary condition function
		"""
        self.dim = dim
        self.Lz = dim[-1]

        if not os.path.isfile(box_mesh_name(dim, res)+".h5"):
            print "Creating a mesh ..."
            DenserBox(dim, res) # create a mesh
        
        # load mesh
        comm = mpi_comm_world()
        f = HDF5File(comm, box_mesh_name(dim, res)+".h5", 'r')
        self.mesh = Mesh()
        f.read(self.mesh, 'mesh', False)
        
        if pb == None:
            V = VectorFunctionSpace(self.mesh, "CG", orders[0], dim=len(dim))  # velocity
            W = FunctionSpace(self.mesh, "CG", orders[1], 1)  # pressure
            X = FunctionSpace(self.mesh, "CG", orders[2], 1)  # temperature

        else:                            
            V = VectorFunctionSpace(self.mesh, "CG", orders[0], dim=len(dim), constrained_domain=pb)  # velocity
            W = FunctionSpace(self.mesh, "CG", orders[1], 1, constrained_domain=pb)  # pressure
            X = FunctionSpace(self.mesh, "CG", orders[2], 1, constrained_domain=pb)  # temperature
        
        self.VWX = MixedFunctionSpace([V, W, X])
        
    
    def on_lid(self, Lz):
        
        class on_lid_class(SubDomain):
        
            def inside(self, x, on_boundary):
                return bool(near(x[1], Lz) and on_boundary)
                
        return on_lid_class()

    class on_base(SubDomain):
        
        def inside(self, x, on_boundary):
            return bool(near(x[1], 0) and on_boundary)
        
    class on_side_walls(SubDomain):
        
        def inside(self, x, on_boundary):
            return bool(on_boundary and not (near(x[1], 0) \
            or near(x[1], Lz)))
        
    class on_all_walls(SubDomain):
        
        def inside(self, x, on_boundary):
            return on_boundary
    
    def make_zero(self, direction, sS=0):
        zeros = None
        if len(self.dim) == 2:
            zeros = Constant((0, 0))
        elif len(self.dim) == 3:
            zeros = Constant((0, 0, 0))
            
        if sS == 0:
            return [DirichletBC(self.VWX.sub(0), zeros, direction), ]
        else:
            return [DirichletBC(self.VWX.sub(sS), 0, direction)]
        

def periodicDomain(Lx, Ly, directions):
    if directions == 1:
        
        class along_x(SubDomain):
            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                return bool(near(x[0], 0) and on_boundary)
                
                # Map right boundary (H) to left boundary (G)
            def map(self, x, y):
                y[0] = x[0] - Lx
                y[1] = x[1]

        return along_x()
        
    else:

        # Periodic boundary condition along two horizontal directions
        class along_xy(SubDomain):
            
            def inside(self, x, on_boundary):
                # return True if on one of the two left boundaries 
                # AND NOT on one of the two slave edges
                return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], Lx) and near(x[1], 0)) or 
                (near(x[0], 0) and near(x[1], 0)))) and on_boundary)

            def map(self, x, y):
                if near(x[0], Lx) and near(x[1], Ly):
                    y[0] = x[0] - Lx
                    y[1] = x[1] - Ly
                    y[2] = x[2]
                elif near(x[0], Lx):
                    y[0] = x[0] - Lx
                    y[1] = x[1]
                    y[2] = x[2]
                elif near(x[1], Ly):
                    y[0] = x[0]
                    y[1] = x[1] - Ly
                    y[2] = x[2]
                else:
                    y[0] = -1000
                    y[1] = -1000
                    y[2] = -1000
                    
        return along_xy()

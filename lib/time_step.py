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

from global_variables import *
from mesh_tools import *
from geometry import *
from eqns import *
import os

# Class representing the intial conditions
class constInitCondition_2d(Expression):
    def __init__(self):
        pass
    def eval(self, values, x):
        values[3] = 0.01
            
    def value_shape(self):
        return (4,)
            
# Class representing the intial conditions
class constInitCondition_3d(Expression):
    def __init__(self):
        pass
    def eval(self, values, x):
        values[4] = 0.01
            
    def value_shape(self):
        return (5,)


class timeStep(globalVariables):
	
    def __init__(self, eqObj, t, T, dt=0.01, dt_save=0.01, gv=[], initData=None):
        """
        eqObj     ---- eqns object
        gvObj     ---- global variables object
        dt        ---- time step
        dt_save   ---- time step to save solution
        gv        ---- list of global variables to compute
        initData  ---- initial data file
        """
        
        self.t = t
        self.T = T
        self.dt = dt
        self.dt_save = dt_save
        self.gv = gv
        self.eqObj = eqObj

        self.comm = mpi_comm_world()        
        
        if initData != None:
            # read the latest solution
            f = HDF5File(self.comm, initData, 'r')
            f.read(eqObj.u, 'u/%g'%t)
            f.read(eqObj.p, 'p/%g'%t)
            f.read(eqObj.c, 'c/%g'%t)
        else:
            # set initial conditions
            eqObj.u_.interpolate(constInitCondition_2d())
            eqObj.u0_.interpolate(constInitCondition_2d())
  
    def constant_dt(self, solver, save_as="hdf5", save_uniform=False):
        """
        Use constant dt for time stepping
        
        solver   ---- dolfin solver object
        save_as  ---- format to save
        
            "hdf5" to save/load data
            "xdmf" for quick visualization (not compatible with save_uniform)
            "foldered_hdf5" for large data sets
            
        save_uniform ---- create and save a dataset on a uniform mesh.
        """
        if save_as == "hdf5":
            out_file = HDF5File(self.comm, "output/out.h5", "w")
            if save_uniform:
                out_file_u = HDF5File(self.comm, "output/out_uniform.h5", "w")
        
        elif save_as == "xdmf":
            u_file = XDMFFile(self.comm, "output/u.xdmf")
            c_file = XDMFFile(self.comm, "output/c.xdmf")
            # flush output immediately
            u_file.parameters["flush_output"] = True
            c_file.parameters["flush_output"] = True
            # don't rewrite
            u_file.parameters["rewrite_function_mesh"] = False
            c_file.parameters["rewrite_function_mesh"] = False
            
        t, T = self.t, self.T
        if MPI.process_number() == 0:
            print "Time stepping with a constant dt=%g"%self.dt
            if not os.path.isfile(box_mesh_name(dim, res)+".h5"):
                if MPI.process_number() == 0:
                    print "creating file output/glob.d"
                os.system("mkdir -p output/")
                os.system("touch output/glob.d")
            glob_file = open("output/glob.d", 'a')
        
        while t < T:
            t += self.dt
            if MPI.process_number() == 0:
                print "t=%g out of T=%g"%(t, T)
            self.eqObj.u0_.vector()[:] = self.eqObj.u_.vector()
            solver.solve()
            gv_ = {func.__name__:func() for func in self.gv}

            if MPI.process_number() == 0:
                print "Computed global variables ..."
                print "System State:  ",
                print ([i + ": %g"%gv_[i] for i in gv_])
            
                # Save global variables in glob.d
                glob_file.write("".join(str(gv_[i])+"  " for i in gv_)+"\n")

            u, p, c = self.eqObj.u_.split()[:]            
            
            if save_uniform:
                mesh1 = self.eqObj.boxObj.mesh
                dim = self.eqObj.boxObj.dim
                res = self.eqObj.boxObj.res
                
                if not os.path.isfile(box_mesh_name(dim, res)+".h5"):
                    print "Creating a uniform mesh ..."
                    DenserBox(dim, res) # create a mesh
        
                # load mesh
                comm = mpi_comm_world()
                f = HDF5File(comm, box_mesh_name(dim, res)+".h5", 'r')
                mesh2 = Mesh()
                f.read(mesh2, 'mesh', False)        
                
                u1 = interpolate_to_mesh(u, mesh1, mesh2)
                p1 = interpolate_to_mesh(p, mesh1, mesh2)
                c1 = interpolate_to_mesh(c, mesh1, mesh2)
                
            if save_as == "hdf5" and t % self.dt_save < 1e-14:
                # save in hdf5 format
                out_file.write(u, "velocity/%g"%t)
                out_file.write(p, "pressure/%g"%t)
                out_file.write(c, "temperature/%g"%t)
                
                if save_uniform:
                    out_file_u.write(u1, "velocity/%g"%t)
                    out_file_u.write(p1, "pressure/%g"%t)
                    out_file_u.write(c1, "temperature/%g"%t)

            elif save_as == "xdmf" and t % self.dt_save < 1e-14:
                u_file << u, t
                c_file << c, t
                
            elif save_as == "foldered_hdf5" and t % self.dt_save < 1e-14:
                dir_ = "output/time_%g"%t
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                    
                u_file = HDF5File(self.comm, "%s/u.h5"%dir_, 'w') 
                p_file = HDF5File(self.comm, "%s/p.h5"%dir_, 'w') 
                c_file = HDF5File(self.comm, "%s/c.h5"%dir_, 'w')
                u_file.write(u, "velocity")
                p_file.write(p, "pressure")
                c_file.write(c, "temperature")
                
                if save_uniform:
                    u_file_u = HDF5File(self.comm, "%s/u_uniform.h5"%dir_, 'w') 
                    p_file_u = HDF5File(self.comm, "%s/p_uniform.h5"%dir_, 'w') 
                    c_file_u = HDF5File(self.comm, "%s/c_uniform.h5"%dir_, 'w')
                    u_file_u.write(u1, "velocity")
                    p_file_u.write(p1, "pressure")
                    c_file_u.write(c1, "temperature")
                    

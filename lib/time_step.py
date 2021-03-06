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
        values[0] = 0.0001*sin(pi*x[1])*cos(x[0])
        values[1] = 0.0001*sin(pi*x[1])*sin(x[0])
            
    def value_shape(self):
        return (4,)
            
# Class representing the intial conditions
class constInitCondition_3d(Expression):
    def __init__(self):
        pass
    def eval(self, values, x):
        values[4] = 0.01*(1-sin(2*pi*x[2]))
            
    def value_shape(self):
        return (5,)


class timeStep(globalVariables):
	
    def __init__(self, comm, eqObj, gv=[], initData=None, out_folder = None, save_as='hdf5', save_uniform=False):
        """
        eqObj     ---- eqns object
        gvObj     ---- global variables object
        dt        ---- time step
        dt_save   ---- time step to save solution
        gv        ---- list of global variables to compute
        initData  ---- initial data file
        """
        
        self.gv = gv
        self.eqObj = eqObj
        self.comm = comm

        if initData != None:
            # read the latest solution
            f = HDF5File(self.comm, initData, 'r')
            f.read(eqObj.u, 'u/%g'%t)
            f.read(eqObj.p, 'p/%g'%t)
            f.read(eqObj.c, 'c/%g'%t)
        else:
            # set initial conditions
            if eqObj.dim == 2:
                eqObj.u_.interpolate(constInitCondition_2d())
                eqObj.u0_.interpolate(constInitCondition_2d())
                eqObj.u00_.interpolate(constInitCondition_2d())
            elif eqObj.dim == 3:
                eqObj.u_.interpolate(constInitCondition_3d())
                eqObj.u0_.interpolate(constInitCondition_3d())
                eqObj.u00_.interpolate(constInitCondition_3d())

        self.out_dir_ = 'output/'
        if out_folder != None:
            self.out_dir = out_folder + '/'

        self.save_as = save_as
        self.save_uniform = save_uniform
                
  
    def constant_dt(self, solver, t, T, dt, dt_save):
        """
        Use constant dt for time stepping
        
        solver   ---- dolfin solver object
        save_as  ---- format to save
        
            "hdf5" to save/load data
            "xdmf" for quick visualization (not compatible with save_uniform)
            "foldered_hdf5" for large data sets
            
        save_uniform ---- create and save a dataset on a uniform mesh.
        """        
       
        save_as = self.save_as
        save_uniform = self.save_uniform
        out_dir = self.out_dir

        if save_as == "hdf5":
            out_file = HDF5File(self.comm, out_dir+"out.h5", "w")
            if save_uniform:
                out_file_u = HDF5File(self.comm, out_dir+"out_uniform.h5", "w")
        
        elif save_as == "xdmf":
            u_file = XDMFFile(self.comm, out_dir+"u.xdmf")
            c_file = XDMFFile(self.comm, out_dir+"c.xdmf")
            # flush output immediately
            u_file.parameters["flush_output"] = True
            c_file.parameters["flush_output"] = True
            # don't rewrite
            u_file.parameters["rewrite_function_mesh"] = False
            c_file.parameters["rewrite_function_mesh"] = False
            
        if MPI.rank(self.comm) == 0:
            print "Time stepping with a constant dt=%g"%dt
            if not os.path.isfile("%sglob.d"%out_dir):
                if MPI.rank(self.comm) == 0:
                    print "creating file glob.d"
                os.system("mkdir -p %s"%out_dir)
                os.system("touch %sglob.d"%out_dir)
            glob_file = open("%sglob.d"%out_dir, 'a', 0)
        
        while t < T:
            t += dt
            if MPI.rank(self.comm) == 0:
                print "t=%g out of T=%g"%(t, T)
                
            self.eqObj.u00_.vector()[:] = self.eqObj.u0_.vector()
            self.eqObj.u0_.vector()[:] = self.eqObj.u_.vector()
            solver.solve()
            gv_ = {func.__name__:func() for func in self.gv}

            if MPI.rank(self.comm) == 0:
                print "Computed global variables ..."
                print "System State:  ",
                print ([i + ": %g"%gv_[i] for i in gv_])
            
                # Save global variables in glob.d
                glob_file.write(str(t) + " " + "".join(str(gv_[i])+"  " for i in gv_)+"\n")

            u, p, c = self.eqObj.u_.split()[:]            
            
            if save_uniform:
                mesh1 = self.eqObj.boxObj.mesh
                dim = self.eqObj.boxObj.dim
                res = self.eqObj.boxObj.res
                
                if not os.path.isfile(box_mesh_name(dim, res)+".h5"):
                    print "Creating a uniform mesh ..."
                    DenserBox(comm, dim, res) # create a mesh
        
                # load mesh
                f = HDF5File(self.comm, box_mesh_name(dim, res)+".h5", 'r')
                mesh2 = Mesh()
                f.read(mesh2, 'mesh', False)        
                
                u1 = interpolate_to_mesh(u, mesh1, mesh2)
                p1 = interpolate_to_mesh(p, mesh1, mesh2)
                c1 = interpolate_to_mesh(c, mesh1, mesh2)
                
            if save_as == "hdf5" and t % dt_save < 1e-14:
                # save in hdf5 format
                out_file.write(u, "velocity/%g"%t)
                out_file.write(p, "pressure/%g"%t)
                out_file.write(c, "temperature/%g"%t)
                
                if save_uniform:
                    out_file_u.write(u1, "velocity/%g"%t)
                    out_file_u.write(p1, "pressure/%g"%t)
                    out_file_u.write(c1, "temperature/%g"%t)

            elif save_as == "xdmf" and t % dt_save < 1e-14:
                u_file << u, t
                c_file << c, t
                
            elif save_as == "foldered_hdf5" and t % dt_save < 1e-14:
                dir_ = out_dir+"time_%g"%t
                if MPI.rank(self.comm) == 0:
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
                    

from eqns import *
from geometry import *
from global_variables import *
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
  
    def constant_dt(self, solver, save_as="hdf5"):
        """
        Use constant dt for time stepping
        
        solver   ---- dolfin solver object
        save_as  ---- format to save
        
            "hdf5" to save/load data
            "xdmf" for visualization
            "foldered_hdf5" for large data sets
        """
        if save_as == "hdf5":
            out_file = HDF5File(self.comm, "output/out.h5", "w")
        
        elif save_as == "xdmf":
            u_file = XDMFFile(self.comm, "output/u.xdmf")
            c_file = XDMFFile(self.comm, "output/c.xdmf")
            
            # flush output immediately
            u_file.parameters["flush_output"] = True
            u_file.parameters["flush_output"] = True
            
            # don't rewrite
            u_file.parameters["rewrite_function_mesh"] = False
            c_file.parameters["rewrite_function_mesh"] = False
            
        t, T = self.t, self.T
        if MPI.process_number() == 0:
            print "Time stepping with a constant dt=%g"%self.dt  
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
            
            if save_as == "hdf5" and t % self.dt_save < 1e-14:
                # save in hdf5 format
                out_file.write(self.eqObj.u_.split()[0], "u/%g"%t)
                out_file.write(self.eqObj.u_.split()[1], "p/%g"%t)
                out_file.write(self.eqObj.u_.split()[2], "c/%g"%t)

            elif save_as == "xdmf" and t % self.dt_save < 1e-14:
                u_file << self.eqObj.u_.split()[0], t
                c_file << self.eqObj.u_.split()[1], t
                
            elif save_as == "foldered_hdf5" and t % self.dt_save < 1e-14:
                dir_ = "output/time_%g"%t
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
                u_file = HDF5File(self.comm, "%s/u.h5"%dir_, 'w') 
                p_file = HDF5File(self.comm, "%s/p.h5"%dir_, 'w') 
                c_file = HDF5File(self.comm, "%s/c.h5"%dir_, 'w')
                u_file.write(self.eqObj.u_.split()[0], "velocity")
                p_file.write(self.eqObj.u_.split()[1], "pressure")
                c_file.write(self.eqObj.u_.split()[2], "temperature")
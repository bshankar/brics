import sys
import os
# append lib/ to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from eqns import *
from geometry import *
from global_variables import *
from time_step import *

##############################################################
# petsc options
PETScOptions.set("pc_factor_mat_solver", "mumps");
PETScOptions.set("pc_type", "lu")

set_log_level(ERROR)

# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

###############################################################

Lx, Lz = 1, 1  # DONT modify Lz!

pb = periodicDomain(Lx, Lz, 1)
b = box((Lx, Lz), (64, 64), pb=pb)  # geometry
rbc = RayleighBenard(b, 2000, 1.0)  # eqns
wf = rbc.cn()  # crank nicholson weak form

# initial conditions
glob = globalVariables(b, rbc)
ts = timeStep(rbc, 0, 0.2, 0.01, 0.02, gv=[glob.Ey, glob.Eth, glob.Nu])
bcs = b.make_zero(b.on_base())    + b.make_zero(b.on_lid(Lz)) + \
      b.make_zero(b.on_base(), 2) + b.make_zero(b.on_lid(Lz), 2)

################################################################

J = derivative(wf, rbc.u_, rbc.du_)
problem = NonlinearVariationalProblem(wf, rbc.u_, bcs, J)
solver = NonlinearVariationalSolver(problem)

#Solver parameters
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-6
prm['newton_solver']['relative_tolerance'] = 1E-5
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0

ts.constant_dt(solver, "foldered_hdf5")

################################################################
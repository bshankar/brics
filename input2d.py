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
#PETScOptions.set("pc_factor_mat_solver", "mumps");
#PETScOptions.set("pc_type", "lu")

set_log_level(ERROR)
comm = mpi_comm_world()

# Form compiler options
parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

###############################################################

Lx, Lz = 2.828, 1.0  # DONT modify Lz!
# Lx Ly Lz must be set at the top of lib/geometry.py!

pb = periodicDomain(Lx, Lz, 1)
b = box(comm, (Lx, Lz), (128, 64), (1, 1.2), orders=(2, 1, 1), pb=pb)  # geometry
rbc = RayleighBenard(b, 657*30, 6.8, scaling=('large', 'small'))  # eqns
U_am2, P_am2, C_am2 = rbc.UPC('am2')
wf_am2 = rbc.linear_terms(U_am2, P_am2, C_am2) + rbc.nonlinear_terms('am2')
# initial conditions
glob = globalVariables(b, rbc)

try:
    folder = sys.argv[1]
except:
    folder = 'output/'
ts = timeStep(comm, rbc, gv=[glob.Ey, glob.Eth, glob.Nu], save_as='foldered_hdf5', out_folder=folder)

# set the boundary conditions
temp_bcs =  b.make_zero(b.on_base(), 2) + b.make_zero(b.on_lid(), 2)
no_slip_bcs = b.make_zero(b.on_base())    + b.make_zero(b.on_lid()) 
# choose the set of boundary conditions
#bcs = no_slip_bcs + temp_bcs 
bcs = b.free_slip_z() + temp_bcs 

################################################################

J_am2 = derivative(wf_am2, rbc.u_, rbc.du_)
problem_am2 = NonlinearVariationalProblem(wf_am2, rbc.u_, bcs, J_am2)
solver_am2 = NonlinearVariationalSolver(problem_am2)

#Solver parameters
prm_am2 = solver_am2.parameters
prm_am2['newton_solver']['linear_solver'] = 'lu'
prm_am2['newton_solver']['absolute_tolerance'] = 1E-8
prm_am2['newton_solver']['relative_tolerance'] = 1E-7
prm_am2['newton_solver']['maximum_iterations'] = 25
prm_am2['newton_solver']['relaxation_parameter'] = 1.0

ts.constant_dt(solver_am2, 0, 10, 0.01, 1.0)

################################################################

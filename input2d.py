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
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

###############################################################

Lx, Lz = 2.02, 1.0  # DONT modify Lz!
# Lx Ly Lz must be set at the top of lib/geometry.py!

pb = periodicDomain(Lx, Lz, 1)
b = box(comm, (Lx, Lz), (128, 64), orders=(2, 1, 1), pb=pb)  # geometry
rbc = RayleighBenard(b, 10000, 0.71, scaling=('large', 'small'))  # eqns
wf = rbc.cn()  # crank nicholson weak form

# initial conditions
glob = globalVariables(b, rbc)
ts = timeStep(comm, rbc, 0, 10.0, 0.01, 0.2, gv=[glob.Ey, glob.Eth, glob.Nu])

# set the boundary conditions
temp_bcs =  b.make_zero(b.on_base(), 2) + b.make_zero(b.on_lid(), 2)
no_slip_bcs = b.make_zero(b.on_base())    + b.make_zero(b.on_lid()) 

# choose the set of boundary conditions
bcs = no_slip_bcs + temp_bcs 

################################################################

J = derivative(wf, rbc.u_, rbc.du_)
problem = NonlinearVariationalProblem(wf, rbc.u_, bcs, J)
solver = NonlinearVariationalSolver(problem)

#Solver parameters
prm = solver.parameters
prm['newton_solver']['linear_solver'] = 'lu'
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['relaxation_parameter'] = 1.0

#prm = solver.parameters
#prm['nonlinear_solver'] = 'snes' 
#prm_snes   = prm['snes_solver']
#prm_snes['linear_solver'] = 'gmres'
#prm_snes['absolute_tolerance'] = 1e-7
#prm_snes['relative_tolerance'] = 1e-6
#prm_snes['solution_tolerance'] = 1e-8
#prm_snes['report'] = False
#prm_krylov = prm_snes['krylov_solver']
#prm_krylov['absolute_tolerance'] = 1e-7
#prm_krylov['relative_tolerance'] = 1e-5
#prm_krylov['gmres']['restart'] = 100

#plot(b.mesh)
#interactive()
ts.constant_dt(solver, "xdmf", True)

################################################################

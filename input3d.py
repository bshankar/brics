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

set_log_level(PROGRESS)
comm = mpi_comm_world()

# Form compiler options
parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = False
parameters["linear_algebra_backend"] = "PETSc"
parameters["allow_extrapolation"] = True

###############################################################
Lx, Ly, Lz = 2.02, 2.02, 1  # DONT modify Lz!

pb = periodicDomain(Lx, Ly, 2)
b = box(comm, (Lx, Ly, Lz), (64, 64, 32), (0, 0, 1.2), pb=pb, orders=(1, 1, 1))  # geometry
rbc = RayleighBenard(b, 2800, 1.0, dim=3, scaling=('large', 'small'))  # eqns
U, P, C = rbc.UPC('am2')
wf = rbc.linear_terms(U, P, C) + rbc.nonlinear_terms('am2')  # crank nicholson weak form

# initial conditions
glob = globalVariables(b, rbc)
ts = timeStep(comm, rbc, gv=[glob.Ey, glob.Eth, glob.Nu])
temp_bcs = b.make_zero(b.on_base(), 2) + b.make_zero(b.on_lid(), 2)
bcs = b.free_slip_z() + temp_bcs



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

#a, L = system(wf)
#problem = LinearVariationalProblem(a, L, rbc.u_, bcs=bcs)
#solver = LinearVariationalSolver(problem)

ts.constant_dt(solver, 0, 10, 0.01, 0.01, "xdmf", True)

################################################################

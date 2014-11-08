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
from geometry import *

class RayleighBenard():
    
    def __init__(self, boxObj, Ra, Pr, dt=0.01, dim=2, scaling=("large", "small")):
        """
        Ra  ---- Rayleigh number
        Pr  ---- Prandl number
        dt  ---- time step
        dim ---- number of dimensions
        """
        
        # set the parameters
        self.scaling, self.dim = scaling, dim
        self.Ra, self.Pr, self.dt, self.I = Ra, Pr, dt, Identity(dim)
        
        # create functions
        # Define trail and test functions
        self.du_ = TrialFunction(boxObj.VWX)
        self.v_u, self.v_p, self.v_c = TestFunctions(boxObj.VWX)

        # define functions
        self.u_ = Function(boxObj.VWX)  # current solution
        self.u0_ = Function(boxObj.VWX)  # solution from previous step
        self.u00_ = Function(boxObj.VWX)
		
        # Split mixed functions
        self.du, self.dp, self.dc = split(self.du_)
        self.u, self.p, self.c = split(self.u_)
        self.u0, self.p0, self.c0 = split(self.u0_)
        self.u00, self.p00, self.c00 = split(self.u00_)
        self.boxObj = boxObj
    
    def UPC(self, method='am2'):    
        U, P, C = None, None, None
        u, u0, u00 = self.u, self.u0, self.u00
        p, p0, p00 = self.p, self.p0, self.p00
        c, c0, c00 = self.c, self.c0, self.c00
        if method == 'ab2':
            U = 0.5*(3*u0 - u00)
            P = 0.5*(3*p0 - p00)
            C = 0.5*(3*c0 - c00)
        elif method == 'am2':
            U = 0.5*(u0 + u)
            P = 0.5*(p0 + p)
            C = 0.5*(c0 + c)
        elif method == 'am3':
            U = (5*u + 8*u0 - u00)/12.0
            P = (5*p + 8*p0 - p00)/12.0
            C = (5*c + 8*c0 - c00)/12.0
        return U, P, C
        
    def linear_terms(self, U, P, C, div=div, grad=grad, dx=dx):
        # all linear terms of RBC equations
        Ra, Pr, dT, I   = self.Ra, self.Pr, self.dt, self.I
        u, u0, u00 = self.u, self.u0, self.u00
        p, p0, p00 = self.p, self.p0, self.p00
        c, c0, c00 = self.c, self.c0, self.c00
        v_u, v_p, v_c = self.v_u, self.v_p, self.v_c

        zi = self.dim-1 # index of z component
        Pscaling, Uscaling = self.scaling
        
        if Pscaling == 'large':
            if Uscaling == 'small':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(P, div(v_u))*dx - \
                     Ra*Pr*inner(C*I[:, zi], v_u)*dx +  \
                     Pr*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx
                     
                L1 = (1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     inner(grad(C), grad(v_c))*dx

            elif Uscaling == 'large':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(P, div(v_u))*dx - \
                     inner(C*I[:, zi], v_u)*dx +  \
                     sqrt(Pr/Ra)*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = (1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     1/sqrt(Pr*Ra)*inner(grad(C), grad(v_c))*dx
                     
        elif Prscaling == 'small':
            if Uscaling == 'small':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(P, div(v_u))*dx - \
                     Ra*inner(C*I[:, zi], v_u)*dx +  \
                     inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = Pr*(1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     inner(grad(C), grad(v_c))*dx

            elif Uscaling == 'large':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(P, div(v_u))*dx - \
                     Pr*inner(C*I[:, zi], v_u)*dx +  \
                     sqrt(Pr/Ra)*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = Pr*(1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     sqrt(Pr/Ra)*inner(grad(C), grad(v_c))*dx
                     
        return L0 + L1
		
    def nonlinear_terms(self, method='ab2'):
        u, u0, u00 = self.u, self.u0, self.u00
        p, p0, p00 = self.p, self.p0, self.p00
        c, c0, c00 = self.c, self.c0, self.c00
        v_u, v_p, v_c = self.v_u, self.v_p, self.v_c
        Pscaling, Uscaling = self.scaling
        L0 = None
        L1 = None
        if method == 'ab1':
            L0 = inner(dot(u0, grad(u0)), v_u)*dx
            L1 = inner(dot(u0, grad(c0)), v_c)*dx
        elif method == 'ab2':
            L0 = inner(1.5*dot(u0, grad(u0))  - 0.5*dot(u00, grad(u00)), v_u)*dx
            L1 = inner(1.5*(dot(u0, grad(c0)) - 0.5*dot(u00, grad(c00))), v_c)*dx
        elif method == 'am2':
            L0 = inner(0.5*(dot(u0, grad(u0)) + dot(u, grad(u))), v_u)*dx
            L1 = inner(0.5*(dot(u0, grad(c0)) + dot(u, grad(c))), v_c)*dx
        elif method == 'am3':
            L0 = inner(1.0/12*(5*dot(u, grad(u)) + 8*dot(u0, grad(u0)) - dot(u00, grad(u00))), v_u)*dx
            L1 = inner(1.0/12*(5*dot(u, grad(c)) + 8*dot(u0, grad(c0)) - dot(u00, grad(c00))), v_c)*dx
        
        if Pscaling == 'small':
            L1 *= self.Pr
            
        return L0 + L1 

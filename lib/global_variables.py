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


from geometry import *
from eqns import *

class globalVariables:
    def __init__(self, geObj, eqObj, scales=("large", "small")):
        self.Pscaling = scales[0]
        self.Uscaling = scales[1]
        self.eqObj = eqObj
        
        # set some variables from eqObj
        self.Pr   = self.eqObj.Pr
        self.Ra   = self.eqObj.Ra
        self.u   = self.eqObj.u
        self.c   = self.eqObj.c
        self.I   = self.eqObj.I
        self.dim = self.eqObj.dim
        
        self.mesh = geObj.mesh
        self.vol = assemble(Constant(1)*dx(self.mesh))

    def new(self):
        """
        Constant new
        """
        if self.Uscaling == "small":
            return self.Pr
            
        return sqrt(self.Pr/self.Ra)
                
    def kappa(self):
        """
        Constant kappa
        """
        if self.Uscaling == "small":
            return 1.0
            
        return 1/sqrt(self.Ra*self.Pr)

    def Ewth(self):
        """
        Nusselt number helper function
        """
        u3 = dot(self.u, self.I[:,self.dim-1])
        Ewth_ = project(u3*self.c)
        return assemble(Ewth_*dx)/self.vol

    def Nu(self):
        """
        Compute Nusselt Number            
        """
        if self.Pscaling == 'large':
            if self.Uscaling == 'small':
                return 1 + self.Ewth()
            elif self.Uscaling == 'large':
                return 1 + sqrt(self.Ra*self.Pr)*self.Ewth()
        elif self.Pscaling == 'small':
            if self.Uscaling == 'small':
                return 1 + self.Pr**2*self.Ewth()
            elif self.Uscaling == 'large':
                return 1 + self.Pr*sqrt(self.Ra*self.Pr)*self.Ewth()

    def Ey(self):
        """
        Compute the K.E
        """
        Ey_ = project(dot(self.u, self.u))        
        return assemble(Ey_*dx)/self.vol
        
    def Eth(self):
        """
        Compute the T.E
        """
        Eth_ = project(dot(self.c, self.c))
        return assemble(Eth_*dx)/self.vol

    def dissipationU(self):
        """
        Compute the dissipation rate of K.E
        """
        dU_ = project(dot(curl(self.u), curl(self.u)))
        return assemble(dU_*dx)/self.vol

    def dissipationC(self):
        """
        Compute the dissipation rate of thermal energy
        """
        dC_ = project(dot(grad(self.c), grad(self.c)))
        return assemble(dC_*dx)/self.vol
            
    def Re(self):
        return sqrt(self.Ey())/self.new()
            
    def Re_lambda(self):
        return self.Ey()*sqrt(15/self.dissipationU())
                

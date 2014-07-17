from geometry import *
from eqns import *

class globalVariables:
    def __init__(self, geObj, eqObj, scales=("small", "small")):
        self.Uscaling = scales[0]
        self.Pscaling = scales[1]
        self.eqObj = eqObj
        
        # set some variables from eqObj
        self.P   = self.eqObj.P
        self.R   = self.eqObj.R
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
            return self.P
            
        return sqrt(self.P/self.R)
                
    def kappa(self):
        """
        Constant kappa
        """
        if self.Uscaling == "small":
            return 1.0
            
        return 1/sqrt(self.R*self.P)

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
                return 1 + sqrt(self.R*self.P)*self.Ewth()
        elif self.Pscaling == 'small':
            if self.Uscaling == 'small':
                return 1 + self.P**2*self.Ewth()
            elif self.Uscaling == 'large':
                return 1 + self.P*sqrt(self.R*self.P)*self.Ewth()

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
                
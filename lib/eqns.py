from dolfin import *
from geometry import *

class RayleighBenard():
    
    def __init__(self, boxObj, R, P, dt=0.01, dim=2, scaling=("small", "small")):
        """
        R   ---- Rayleigh number
        P   ---- Prandl number
        dt  ---- time step
        dim ---- number of dimensions
        """
        
        # set the parameters
        self.scaling = scaling
        self.R = R
        self.P = P
        self.dt = dt
        self.dim = dim
        self.I = Identity(dim)
        
        # create functions
        # Define trail and test functions
        self.du_ = TrialFunction(boxObj.VWX)
        self.v_u, self.v_p, self.v_c = TestFunctions(boxObj.VWX)

        # define functions
        self.u_ = Function(boxObj.VWX)  # current solution
        self.u0_ = Function(boxObj.VWX)  # solution from previous step

        # Split mixed functions
        self.du, self.dp, self.dc = split(self.du_)
        self.u, self.p, self.c = split(self.u_)
        self.u0, self.p0, self.c0 = split(self.u0_)
        
        self.U = 0.5*(self.u + self.u0)
        self.C = 0.5*(self.c + self.c0)
        
        
    def cn(self, div=div, grad=grad, dx=dx, nlt=1):
        """
        Return the weak statements of the eqns governing Rayleigh Benard convection.
        grad ---- custom gradient function
        div ---- custom divergence function
        dx  ---- Volume element
        
        """
        
        R = self.R
        P = self.P
        dT = self.dt
        I = self.I
        u = self.u
        u0 = self.u0
        U = self.U
        c = self.c
        c0 = self.c0
        C = self.C
        p = self.p
        v_u = self.v_u
        v_c = self.v_c
        v_p = self.v_p
        
        zi = self.dim-1 # index of z component
        Pscaling, Uscaling = self.scaling        

        if Pscaling == 'large':
            if Uscaling == 'small':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(p, div(v_u))*dx - \
                     R*P*inner(c*I[:, zi], v_u)*dx +  \
                     P*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx
                     
                L1 = (1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     inner(grad(C), grad(v_c))*dx

            elif Uscaling == 'large':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(p, div(v_u))*dx - \
                     inner(c*I[:, zi], v_u)*dx +  \
                     sqrt(P/R)*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = (1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     1/sqrt(P*R)*inner(grad(C), grad(v_c))*dx

            if nlt:
                # enable the non linear terms
                L0 = L0 + inner(0.5*(dot(u0, grad(u0)) + dot(u, grad(u))), v_u)*dx
                L1 = L1 + inner(0.5*(dot(u0, grad(c0)) + dot(u, grad(c))), v_c)*dx

        if Pscaling == 'small':
            if Uscaling == 'small':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(p, div(v_u))*dx - \
                     R*inner(c*I[:, zi], v_u)*dx +  \
                     inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = P*(1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     inner(grad(C), grad(v_c))*dx

            elif Uscaling == 'large':
                L0 = (1. /dT)*inner(u - u0, v_u)*dx - \
                     inner(p, div(v_u))*dx - \
                     P*inner(c*I[:, zi], v_u)*dx +  \
                     sqrt(P/R)*inner(grad(U), grad(v_u))*dx + \
                     inner(div(U), v_p)*dx

                L1 = P*(1./dT)*inner(c - c0, v_c)*dx - \
                     inner(U[zi], v_c)*dx + \
                     sqrt(P/R)*inner(grad(C), grad(v_c))*dx

            if nlt:
                # enable the non linear terms
                L0 = L0 + inner(0.5*(dot(u0, grad(u0)) + dot(u, grad(u))), v_u)*dx
                L1 = L1 + P*inner(0.5*(dot(u0, grad(c0)) + dot(u, grad(c))), v_c)*dx

        return L0 + L1
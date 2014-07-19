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


import h5py
import numpy as np
import matplotlib.pyplot as plt


def rbc_fig2d(u, c, mesh):
    """
    u    ---  velocity field (for a quiver plot)
    c    ---  temperature field (scalar plot)
    """
    
    plt.figure()
    X, Y = mesh.coordinate()[:]
    Q = plt.quiver(X, Y, u)
    qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',
               fontproperties={'weight': 'bold'})
    l,r,b,t = axis()
    dx, dy = r-l, t-b
    axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])
    plt.title("")
    return plt


plt.show()

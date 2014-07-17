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
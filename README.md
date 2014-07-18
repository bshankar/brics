*Brics**
=========
An Incompressible Rayleigh Benard Equation solver that uses FEniCS with
with PETSc backend that aims to be efficient and parallel. 
BRICS is written in python 2.7

Features:
=========
1. Implicit time stepping with adaptivity accurate to O(dt^2)
2. Generates Non-uniform meshes that are denser towards corners  
3. Parallel execution on CPU/GPU due to FEniCS, PETSc 
4. Object oriented code makes creating new domains and defining equations in curved geometries easy.
5. No-slip/free-slip or periodic boundary conditions

Dependencies:
=============
* python 2.7.xx
* PETSc
* FEniCS (1.4 and above)
* Numpy
* Matplotlib


Running the code:
=================

.. code:: bash

    $ git clone https://bshankar@bitbucket.org/bshankar/brics.git
    $ python2 input.py   # input file

Terms of Use
============
Brics is free, and distributed under the GNU General Public License (GPL). Essentially, this means that you are free to do almost exactly what you want with the program, including distributing it among your friends, making it available for download from your web site, selling it (either by itself or as part of some bigger software package), or using it as the starting point for a software project of your own.

The only real limitation is that whenever you distribute Brics in some way, you must always include the full source code, or a pointer to where the source code can be found. If you make any changes to the source code, these changes must also be made available under the GPL.

For full details, read the copy of the GPL found in the file named Copying.txt

Useful links:
=============
* `FEniCS <http://fenicsproject.org/>`_
* Incompressible Rayleigh benard equations and the Boundary conditions
* `tests <https://bitbucket.org/bshankar/brics/wiki/Verify>`_ BRICS passed so far.

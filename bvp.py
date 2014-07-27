from __future__ import division
import numpy.linalg as la
import numpy as np
from gmres import my_gmres_e


def solve_bvp(mesh, p, q, r, ua, ub, iter_count=False):
    r"""Solve the boundary value problem

    .. math::

        u''+p(x)u'+q(x)u=r(x),\quad u(a) = u_a,\quad u(b) =u_b

    on *mesh*. Return a vector corresponding to the solution *u*.
    Uses :func:`apply_fredholm_kernel`.

    :arg mesh: A 1D array of nodes in the interval :math:`[a, b]`,
          with the first equal to :math:`a` and the last equal to :math:`b`.
    :arg p, q, r: Functions that accept a vector *x* and
      evaluate the functions $p$, $q$, $r$ at the nodes in *x*.
    """
    
    x = mesh.copy()
    n = len(x)
    a = x[0]
    b = x[-1]
    l = b-a
    
    #Defining the tau function
    def tau(xi): 
        return (1 - (xi-a)/l)
    #Defining the kernel function to evaluate the 2D Kernel mesh    
    def kernel(x):
        n = len(x)
        K = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                t = x[i]; s = x[j];
                if s <= t:
                    K[i, j] = ( -p(t) + (b-t)*q(t) )/l*(a-s)
                else:
                    K[i, j] = (  p(t) + q(t)*(t-a) )/l*(s-b)
        return K

    R = -(q(x)*(tau(x)*ua + (1-tau(x))*ub) + p(x)/l*(-ua+ub) -r(x))
    
    #Calling the kernel function. It is called only once in the solve_bvp, so that
    #the 2D mesh is not built in each iteration of GMRES
    
    K=kernel(x)

    def apply_kernel(a, b, x, kernel, density):
        r"""Return a vector *F* corresponding to

        .. math::

        F(x) = \int_a^b k(x,z) \phi(z) dz

        evaluated at all points of *mesh* using the
        trapezoidal rule.

        :arg mesh: A 1D array of nodes in the interval :math:`[a, b]`,
          with the first equal to :math:`a` and the last equal to :math:`b`.
        :arg kernel: two-argument vectorized callable ``kernel(tgt, source)``
        that evaluates
        :arg density: Values of the density at the nodes of *mesh*.
        """
        n = len(x)
        F = np.zeros(n)
        for i in range (0, n):
            y = kernel[i,:].T*density
            F[i] = np.trapz(y, x=x)
        return F
    
    #Defining A_func to be passed to GMRES
    def A_func(phi):
        return (apply_kernel(a, b, x, K, phi) + phi )
    
    #Calling GMRES to solve for phi   
    phi, num_it = my_gmres_e(A_func, R, tol=1e-10)

    #Defining the kernel for extracting the solution from phi
    def kernel_sol(x):
        n = len(x)
        K = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                t = x[i]; s = x[j];
                if s < t:
                    K[i, j] = (  (b-t) )/l*(a-s)
                else:
                    K[i, j] = ((t-a) )/l*(s-b)
        return K       

    K_sol=kernel_sol(x)

    u=(tau(x)*ua + (1 - tau(x))*ub + apply_kernel(a, b, x, K_sol, phi))

    if iter_count==False:
        print num_it, "Iterations"
    else:
        print "Working...."
    
    if iter_count==True:
        return u,num_it
    else:
        return u
    






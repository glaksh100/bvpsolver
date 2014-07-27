from __future__ import division
from legendre_discr import CompositeLegendreDiscretization
from gmres import my_gmres_e
import numpy as np
import numpy.linalg as la
import timeit 


def apply_kernel(discr, fl, gl, fr, gr, density):
    """
    :arg discr: an instance of
        :class:`legendre_discr.CompositeLegendreDiscretization`
    :arg fl,gl,fr,gr: functions of a single argument
    """
    mesh=discr.nodes.ravel()
    #Evaluating function to be passed to left integral
    left_func = gl(mesh)*density
    #Evaluating function to be passed to right integral
    right_func= gr(mesh)*density

    left_func_reshape=np.reshape(left_func,(discr.nintervals,discr.npoints))
    right_func_reshape=np.reshape(right_func,(discr.nintervals,discr.npoints))

    G_l=discr.left_indefinite_integral(left_func_reshape)
    G_r=discr.right_indefinite_integral(right_func_reshape)
    
    G_l_reshape=G_l.ravel()
    G_r_reshape=G_r.ravel()

    K= fl(mesh)*G_l_reshape + fr(mesh)*G_r_reshape
    
    return K

def solve_bvp(discr, p, q, r, ua, ub, return_time_iter="False"):
    """
    :arg discr: an instance of
        :class:`legendre_discr.CompositeLegendreDiscretization`

    """
    #Starting timer
    start=timeit.default_timer()
    mesh=discr.nodes.ravel()
    n=len(mesh)
    a=discr.intervals[0]
    b=discr.intervals[-1]
    l=b-a
    

    """ only pass flattened nodes to fl_func ,fr_func, gl_func and gr_func functions"""
    def fl_func(x):
        return (-p(x) + (b-x)*q(x))/l
    def fr_func(x):
        return (p(x) + q(x)*(x-a))/l
    
    def gl_func(x):
        return (a - x)
    
    def gr_func(x):
        return (x-b)


    def tau(xi): 
        return (1 - (xi-a)/l)

    """Building RHS of GMRES"""

    R = -(q(mesh)*(tau(mesh)*ua + (1-tau(mesh))*ub) + p(mesh)/l*(-ua+ub) -r(mesh))
    

    """Defines the function which passes to my_gmres_e"""
    def matrix_func(phi): 
        return (apply_kernel(discr,fl_func,gl_func,fr_func,gr_func,phi) + phi)

    phi, num_it = my_gmres_e(matrix_func, R)

    if(return_time_iter=="False"):
        print num_it, "Iterations"
    else:
        print "Working..."

    """Extracting the solution from phi"""
    def fl_sol(x):
        return (b-x)/l
    def fr_sol(x):
        return (x-a)/l

    left_func = gl_func(mesh)*phi
    right_func= gr_func(mesh)*phi

    left_func_reshape=np.reshape(left_func,(discr.nintervals,discr.npoints))
    right_func_reshape=np.reshape(right_func,(discr.nintervals,discr.npoints))

    G_l=discr.left_indefinite_integral(left_func_reshape)
    G_r=discr.right_indefinite_integral(right_func_reshape)
    G_l_reshape=G_l.ravel()
    G_r_reshape=G_r.ravel()

    K= fl_sol(mesh)*G_l_reshape + fr_sol(mesh)*G_r_reshape
    
    u_sol=tau(mesh)*ua + (1-tau(mesh))*ub + (apply_kernel(discr,fl_sol,gl_func,fr_sol,gr_func,phi))
    u_sol_reshape=np.reshape(u_sol,(discr.nintervals,discr.npoints))

    stop=timeit.default_timer()

    time=stop-start
    if return_time_iter=="True":
        return u_sol_reshape, time, num_it
    else:
        return u_sol_reshape
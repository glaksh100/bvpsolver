from __future__ import division
import numpy as np
from fast_bvp import solve_bvp
import matplotlib.pyplot as pt

a = -4
b = 5

def test_poisson(discr):
    def u_true(x):
        return np.sin(5*x)*np.exp(x)

    def r(x):
        return 10*np.exp(x)*np.cos(5*x)-24*np.exp(x)*np.sin(5*x)

    def p(x):
        return 0

    def q(x):
        return 0

    u, t, nit = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b),return_time_iter="True")


    return nit,t


def test_with_q(discr):
    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return 0

    def q(x):
        return 1/(x+7)

    def r(x):
        return np.sin(5*x**2)/(x+7)-100*x**2*np.sin(5*x**2)+10*np.cos(5*x**2)

    u , t, nit= solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b),return_time_iter="True")

    return nit,t


def test_with_p(discr):
    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 0

    def r(x):
        return (
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2)
                )

    u,t,nit = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b),return_time_iter="True")

    return nit,t


def test_full_enchilada(discr):
    def u_true(x):
        return np.sin(5*x**2)

    def p(x):
        return np.cos(x)

    def q(x):
        return 1/(x+7)

    def r(x):
        return (np.sin(5*x**2)/(x+7)
                - 100*x**2*np.sin(5*x**2)
                + 10*x*np.cos(x)*np.cos(5*x**2)+10*np.cos(5*x**2))

    u,t,nit = solve_bvp(discr, p, q, r, ua=u_true(a), ub=u_true(b),return_time_iter="True")

    return nit,t


def estimate_order(f, point_counts):
    n1, n2 = point_counts
    h1, err1 = f(n1)
    h2, err2 = f(n2)

    print "h=%g err=%g" % (h1, err1)
    print "h=%g err=%g" % (h2, err2)

    from math import log
    est_order = (log(err2/err1) / log(h2/h1))
    print "%s: EOC: %g" % (f.__name__, est_order)
    print

    return est_order


if __name__ == "__main__":
    from legendre_discr import CompositeLegendreDiscretization

    for test in [test_poisson, test_with_p, test_with_q, test_full_enchilada]:
        for order in [3, 5 , 7]:
                n=[]
                t=[]
                nit=[]
                for i in range(3,11):
                    intervals = np.linspace(0, 1, 2**i, endpoint=True) * (b-a) + a
                    discr = CompositeLegendreDiscretization(intervals, order)
                    n.append(2**i-1)
                    val_nit,val_t=test(discr)
                    nit.append(val_nit)
                    t.append(val_t)
                pt.figure(1)
                pt.plot(n,nit,label='%d %s' %(order,test.__name__))
                pt.xlabel("Number of subintervals")
                pt.ylabel("Number of iterations")
                pt.legend(loc=2,prop={'size':7})
                pt.figure(2)
                pt.plot(n,t,label='%d %s' %(order,test.__name__))
                pt.xlabel("Number of subintervals")
                pt.ylabel("Computational Time")
                pt.legend(loc=2,prop={'size':7})
    print "Done"
    pt.show()
                

            #assert estimate_order(get_error, [50, 100]) >= order - 0.5
